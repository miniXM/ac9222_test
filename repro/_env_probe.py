
import torch
print("torch              =", torch.__version__)
print("torch.version.cuda  =", torch.version.cuda)
print("cuda available      =", torch.cuda.is_available())
print("gpu count           =", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(i, p.name, "SM", "%d.%d" % (p.major, p.minor),
          "VRAM_GB", round(p.total_memory / 1024**3, 1),
          "SMs", p.multi_processor_count)
print("cudnn               =", torch.backends.cudnn.version())
print("cuda_arch_list      =", torch.cuda.get_arch_list())
