#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/core/Layout.h>
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/headeronly/version.h>
#include <cuda_runtime.h>

#include <array>
#include <optional>

// [AC922 patch 2026-10-02] torch 2.10 stable ABI 差异：
//   1) torch::stable::Tensor 在 2.10 没有 layout() 成员（2.11 才有）。
//      本函数只做 CPU->CUDA 的 UVA 视图，layout 恒为 Strided。
//   2) torch::stable::from_blob 在 2.10 的第 6 个参数是 int64_t storage_offset，
//      没有接受 deleter 的重载（2.11 才有）。这里退化为不带 deleter 的版本。
//      代价：返回的 CUDA 视图不持有 host 内存所有权，调用方需保活源 tensor。

// This function assumes that `cpu_tensor` is a CPU tensor,
// and that UVA (Unified Virtual Addressing) is enabled.
torch::stable::Tensor get_cuda_view_from_cpu_tensor(
    torch::stable::Tensor& cpu_tensor) {
  STD_TORCH_CHECK(cpu_tensor.device().is_cpu(), "Input tensor must be on CPU");

  const auto dtype = cpu_tensor.scalar_type();
  const auto layout = torch::headeronly::Layout::Strided;
  const torch::stable::Device cuda_dev(torch::headeronly::DeviceType::CUDA);

  // handle empty tensor
  if (cpu_tensor.numel() == 0) {
    return torch::stable::empty(cpu_tensor.sizes(), dtype, layout, cuda_dev);
  }

  std::array<StableIValue, 2> is_pinned_stack{
      torch::stable::detail::from(cpu_tensor),
      torch::stable::detail::from(std::nullopt)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::is_pinned", "", is_pinned_stack.data(), TORCH_ABI_VERSION));
  if (torch::stable::detail::to<bool>(is_pinned_stack[0])) {
    // If CPU tensor is pinned, directly get the device pointer.
    void* host_ptr = const_cast<void*>(cpu_tensor.mutable_data_ptr());
    void* device_ptr = nullptr;
    cudaError_t err = cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
    STD_TORCH_CHECK(err == cudaSuccess, "cudaHostGetDevicePointer failed: ",
                    cudaGetErrorString(err));

    return torch::stable::from_blob(device_ptr, cpu_tensor.sizes(),
                                    cpu_tensor.strides(), cuda_dev, dtype);
  }

  // If CPU tensor is not pinned, allocate a new pinned memory buffer.
  torch::stable::Tensor contiguous_cpu = torch::stable::contiguous(cpu_tensor);
  size_t nbytes = contiguous_cpu.numel() * contiguous_cpu.element_size();

  void* host_ptr = nullptr;
  cudaError_t err = cudaHostAlloc(&host_ptr, nbytes, cudaHostAllocMapped);
  if (err != cudaSuccess) {
    STD_TORCH_CHECK(false, "cudaHostAlloc failed: ", cudaGetErrorString(err));
  }

  err = cudaMemcpy(host_ptr, contiguous_cpu.const_data_ptr(), nbytes,
                   cudaMemcpyDefault);
  if (err != cudaSuccess) {
    cudaFreeHost(host_ptr);
    STD_TORCH_CHECK(false, "cudaMemcpy failed: ", cudaGetErrorString(err));
  }

  void* device_ptr = nullptr;
  err = cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
  if (err != cudaSuccess) {
    cudaFreeHost(host_ptr);
    STD_TORCH_CHECK(
        false, "cudaHostGetDevicePointer failed: ", cudaGetErrorString(err));
  }

  return torch::stable::from_blob(device_ptr, contiguous_cpu.sizes(),
                                  contiguous_cpu.strides(), cuda_dev,
                                  contiguous_cpu.scalar_type());
}
