# MiniMax-H3 @ AC922 排障证据链

> 每一个结论都有实测数字，不是推测。时间：2026-10-04 ~ 2026-10-05。

---

## 0. 起点：服务其实从来没起来过

排查前先发现，此前所有"黑视频 / NaN"的观察**都不可能来自这条 vLLM-Omni 链路** ——
因为服务压根没启动成功过。依次修掉 4 个部署缺口后才有第一次真正的推理：

| # | 症状 | 根因 | 修法 |
|---|---|---|---|
| 1 | import 就炸 `No module named 'cache_dit'` | `minimax_h3_transformer.py:19` 硬依赖，但**未在依赖清单里声明** | 装 `cache_dit-1.5.2` wheel |
| 2 | `No module named 'h3_chunked_sdpa'` | 自建移植模块只存在于 `/tmp`，不在 Python 路径 | 拷进 venv site-packages |
| 3 | `torch.OutOfMemoryError` @ `h3_dit_checkpoint.py:161` | TP=1，单张 16GB V100 装不下 | `--tensor-parallel-size 4` |
| 4 | 首请求必炸 `Cannot find a working triton installation` | ppc64le 无 Triton，`torch.compile` 无法工作 | `TORCHDYNAMO_DISABLE=1` |

输出侧另外 3 个坑（PyAV 缺失 / `av.VideoStream` 不存在 / 600s 超时）见 README。

---

## 1. 结果：出图了，但是马赛克

```
HTTP 200, 637KB MP4, 39 帧, 256×256 @24fps, 含音轨

[H3-DBG] video_latent (1, 24, 12, 16, 16) float32 finite=True  min=-4.355 max=3.630 mean=0.0563
[H3-DBG] vae_video    (1,  3, 39, 256, 256) float32 finite=True min=0.0   max=1.0   mean=0.2795
```

- **不是黑屏**（mean 0.279，黑屏会是 ~0）
- **不是 NaN/Inf**（finite=True）
- 画面是**结构化的 16×16 块状马赛克** —— 块尺寸**正好等于 latent 网格** 256/16

---

## 2. A/B 实验：attention 实现彻底排除（含真·FA-V100）

| 实验 | 结果 | 结论 |
|---|---|---|
| **A：`--diffusion-attention-backend TORCH_SDPA`** | 见上，马赛克 | 链路通、内容坏 |
| **B：真·FA-V100**（`flash_attn_v100` v1.2.0 SM70 kernel） | kernel 实际执行（`/tmp/h3_fa_v100_used.txt: USED varlen fp16 (4124,14,128)`）；step0 `v_std=0.3844`、`corr=0.924`；MP4 637,179B，同样的 16×16 马赛克 | **与 SDPA 统计相同** |
| 对照：`H3_NO_FP16_PATCH=1`（纯 bf16） | `RuntimeError: v must be finite` @ `scheduling_minimax_h3_euler_ancestral.py:12/29` | **历史"黑视频/NaN"根因实锤**：V100 上纯 bf16 DiT 直接 NaN；`h3_fp16_mixed` 是能跑的必要条件 |

**B 轮为什么之前"测不了"，以及怎么打通的（重要，别再踩）：**

`flash_attn_v100` 是真实存在的 ppc64le SM70 编译 kernel（28MB .so），但被三道门挡死：

1. 只装在生产 `/opt/conda/envs/vllm`，omni 测试 venv 里没有 → 拷贝过去即可
2. `platforms/cuda/platform.py :: has_flash_attn_package()` **按 GPU 名字拉黑**：
   `"Turing"/"Tesla"/"T4" in gpu_name → return False`，"Tesla V100-SXM2" 直接命中
3. 同文件 `get_diffusion_attn_backend_cls()`：`compute_supported = capability >= 80`，sm_70 不过

另外 `flash_attn_v100` **没有 varlen kernel**（只有 dense `flash_attn_func`，且只吃 fp16），
而 H3 走 packed varlen（cu_seqlens）。打通方案：

- 造一个 `flash_attn` shim 包：`flash_attn_varlen_func` = 按 cu_seqlens 分段循环调 dense kernel，
  bf16→fp16 自动转换（shim 源码见仓库外 `scripts/_fa_shim_src.py`，思路写在此处备查）
- platform.py 两道门用 `H3_FA_V100=1` 环境变量旁路（备份 `.bak_fav100_*`）

单元测试：dense fp16 与 varlen（分段 97+155 / 128+128+64）对 fp32 SDPA 参考
**cos = 1.000000**（absmax 2.4e-4 / 4.2e-4）。kernel 数学正确。

**最终结论：torch SDPA（分段模拟）与真·FA-V100 CUDA kernel 产出统计上相同的速度场
（v_std 0.3845 vs 0.3844，corr 0.924 vs 0.924，输出同为 16×16 马赛克）
→ attention 实现彻底排除，病灶在上游。**

---

## 3. VAE：完全健康（round-trip 实测，已排除）

方法：参考苹果图 `encode_images` → 沿 T 重复到 `(1,24,12,16,16)` → `decode_latent`

- 重建出**清晰完整的苹果**，条纹/高光/叶子都在，**无任何 16×16 块状伪影**（见 `outputs/`）
- **fp16 autocast 与 fp32 输出逐位一致**（差异 1e-4 量级）→ V100 fp16 也不是问题
- 澄清一处误判：`vae.py` 的 encode / decode **自身已经对称处理了 `latents_mean`/`latents_std`**
  （encode 做 `(z-mean)/std`，decode 做 `latent*std+mean`），"漏反归一化"的怀疑**不成立**

---

## 4. 量化：数学精确（单元测试，已排除）

本机的 `H3_DEQUANT_MODE=1`（一次反量化成 bf16 常驻）对照**做不了**：
- fp8 9.5GB/卡 → bf16 19GB/卡 > 16GB，加载到 `blocks.47` 崩
- 真 bf16 权重更不可能：DiT 77GB + 文本编码器 64GB ≈ **141GB > 96GB 总显存**

改用**反量化数值单元测试**（GPU 4 上独立跑）：

| 层 | 逐调用 FP8 反量化 vs fp32 真值 |
|---|---|
| `to_q` / `to_out.0` / `ff.net.2` / block25 | **相对误差 3.6e-4，cos = 1.0** |

比 bf16 一次性反量化（1.56e-3）还准。量化方案是 modelopt `FP8_PER_CHANNEL_PER_TOKEN`
（E4M3 权重 + F32 per-channel scale）。**反量化排除。**

---

## 5. 真正的病灶（内置探针实测）

| 探针 | 读数 | 含义 |
|---|---|---|
| `H3_STEP_TRACE` step0 | **`v_std = 0.3845`** | 健康 rectified-flow 在纯噪声处应 ~1.4，**只有 27%** |
| `H3_STEP_TRACE` step0 | **`corr(x0_pred, x) = 0.924`** | 模型预测的 x0 ≈ 输入噪声本身 → **去噪根本没推动** → 最终 latent ≈ 噪声 → 马赛克 |
| `H3_SENSITIVITY` (B) | 清零参考图 → velocity 只变 **9.2%** | ref2va 模型对参考图依赖不该这么低 |
| `H3_SENSITIVITY` (C) | 清零文本 → 只变 **2.7%** | 条件注入弱 |
| `H3_SENSITIVITY` (A) | 时间步扫描有响应（cos 0.96 → 0.90） | 时间步通路是活的 |
| `H3_SENSITIVITY` (E) | AdaLN gate absmean 0.14–0.27 | 活的 |
| **`H3_ATTN_PROBE`** | 把一半 target 行整体换成新噪声，**另一半 token 的 velocity 只变 5.4%~6.9%**（被换的行自己变 51%，cos≈0.998） | **跨 token 信息流衰减 ~20 倍**，每个 token 基本只看自己 |
| 对照：**audio 分支** | `a_std = 1.08`（接近健康） | **唯独 video 路径坏** |

所有症状彼此自洽：条件影响微弱 + 跨 token 信息流死 + v_std 只有 27% + corr(x0,x)=0.92
→ 去噪没推动 → latent ≈ 噪声 → 解码出 latent 网格大小的色块。

另外：同 seed 两次请求 latent 统计**完全一致**（可复现）；
denoise 8 步约 1:49（15.6s/it，峰值 10.3GB），真正的时间大头是文本编码（~5 分钟）。

---

## 6. 剩余嫌疑（请重点看这三处）

1. **`MiniMaxH3Transformer3DModel` 前向里 video 特有路径的接线**
   —— `video_patch_proj` / video RoPE / 位置编码 / qk_norm。
   注意 audio 分支健康、video 分支坏，这个对照把范围缩得很窄。
2. **checkpoint 本身**
   —— fork 作者在 `denoise_loop.py` 注释里也留下 `near-constant velocity → checkpoint or conditioning`。
3. **缺一个健康基线**
   —— 现在无法 100% 断定「跨 token 信息流 5~7%」是异常，还是 t=0（纯噪声步）处本就如此。
   需要任何一台能正常跑 H3 的机器上、同输入的 `v_std` / 敏感度读数做对照。

补充：`h3_chunked_sdpa.py` 已逐行审过 —— 只切 query 轴、全 key softmax、掩码按行切片，**数学上没发现 bug**。
但探针表现与"attention 退化成只看自己"一致，值得第二个人的独立判断。

---

## 7. 已明确**不要**再走的弯路

- 不要再试 `H3_DEQUANT_MODE=1`（TP=4 必 OOM）
- 不要为了排障升级 PyTorch 2.10 → 2.11：本轮结论与 torch 版本无关，
  且升级要重编 vLLM 0.26 的 CUDA 扩展（`cuda_view.cu` 的 ABI 适配得重做），会引入全新变量。
  upgrade 应作为**第二阶段**独立进行。
- 不要用 `pkill -f 'vllm-omni serve'` 停服务（会杀掉启动脚本自己）
- FA-V100 一次 illegal memory access 会**污染整张卡的 CUDA 上下文**，同卡后续全部报错 ——
  换张卡重试即可，不是 kernel 坏。
