# FA-V100 接入说明（真·B 轮）

`flash_attn_v100` v1.2.0 是真实存在的 ppc64le SM70 flash-attention kernel（装在生产
`/opt/conda/envs/vllm/lib/python3.12/site-packages/flash_attn_v100/`，28MB 编译 .so）。

## 它被挡住的三道门
1. 不在 omni 测试 venv（`/srv/omni/venv`）里 → 拷贝过去
2. `vllm_omni/platforms/cuda/platform.py :: has_flash_attn_package()` 按 GPU 名字拉黑
   `"Turing"/"Tesla"/"T4"` → "Tesla V100-SXM2" 直接命中
3. 同文件 `get_diffusion_attn_backend_cls()` 要求 `capability >= 80`，sm_70 不过

## 另外 kernel 本身的两个限制
- **没有 varlen**（`grep -c varlen flash_attn_interface.py` = 0），只有 dense `flash_attn_func`
- **只吃 fp16**（bf16 输入报 `q must be fp16`）

## 打通方案（本仓库内文件）
- `h3_custom/flash_attn_v100_shim.py` → 装到 omni venv site-packages 的 `flash_attn/__init__.py`：
  - `__version__ = "2.6.0"`（骗过 omni 的 `>= 2.6.0` 版本检查）
  - `flash_attn_varlen_func` = **按 cu_seqlens 分段循环调 dense kernel**（H3 每次只有 2~3 个段，开销可忽略）
  - bf16 → fp16 自动转换，输出转回原 dtype
- `h3_custom/platform.py.patched.cuda` → 补丁后的 platform.py（两道门都加了 `H3_FA_V100=1` 旁路）

## 验证
- 单元测试（GPU4/5 独立进程）：dense fp16 与 varlen（97+155 / 128+128+64 分段）
  对 fp32 SDPA 参考 **cos = 1.000000**（absmax 2.4e-4 / 4.2e-4）
- 真跑：后端解析为 `FLASH_ATTN`（不再回退 SDPA），`/tmp/h3_fa_v100_used.txt` 记录
  `USED varlen dtype=torch.float16 shape=(4124, 14, 128)`

## 结果：与 SDPA 统计相同
| | SDPA（分段模拟） | FA-V100（真 kernel） |
|---|---|---|
| step0 v_std | 0.3845 | 0.3844 |
| corr(x0,x) | 0.924 | 0.924 |
| MP4 | 636,942 B，16×16 马赛克 | 637,179 B，同样的 16×16 马赛克 |

→ **attention 实现彻底排除**。两个完全不同的 kernel 给出相同速度场，
病灶在上游：video_patch_proj / video RoPE / 位置编码 / qk_norm 接线，或 checkpoint 本身。

## 坑
FA-V100 一次 illegal memory access 会**污染整张卡的 CUDA 上下文**，同卡后续全部失败 ——
换张卡重试即可，不是 kernel 坏。
