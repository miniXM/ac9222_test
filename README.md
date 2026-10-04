# MiniMax-H3 @ IBM AC922（ppc64le / V100-SM70）— vLLM-Omni 0.26 移植代码与排障快照

> 用途：把本机上「能让 MiniMax-H3 跑起来但出图是马赛克」的完整代码现场交给人分析。
> 所有内容都是**现役工作树上直接拷出来的真实状态**，不是整理过的理想版本。

---

## 一、一句话现状

**端到端已经跑通**：`POST /v1/videos/sync` 返回 HTTP 200，产出 637KB MP4（39 帧 256×256 @24fps，含音轨）。
但画面是 **16×16 块状马赛克** —— 数值上完全健康（finite、量纲正确、不黑不 NaN），**语义上是噪声**。

详细的证据链、已排除项、剩余嫌疑见 **[ANALYSIS.md](ANALYSIS.md)**。

---

## 二、机器实况（实测，非推断）

```
OS              Debian GNU/Linux 12 (bookworm)
CPU             POWER9 / ppc64le
GPU             6 × Tesla V100-SXM2-16GB  (SM 7.0, 15.8GB each, 80 SMs)  → 96GB 总显存
NVIDIA driver   535.216.03
CUDA Toolkit    12.4
PyTorch         2.10.0  (torch.version.cuda = 12.4, cudnn 8907, arch_list = ['sm_70'])
vLLM            0.26.0+cu124（自编译）
vLLM-Omni       0.26.0（editable install）
Python          3.12
Triton          ❌ 不存在（ppc64le 无官方 wheel）
```

⚠️ **两个容易踩死的坑**

1. `/usr/local/cuda` 软链指向的是 **cuda-11.8**，不是 12.4（`/usr/local/cuda-12.2` 不存在，
   `nvidia-smi` 显示的 `CUDA Version: 12.2` 只是驱动侧能力值）。
   所有 torch / vllm 命令**必须**显式带：
   ```
   export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:/usr/local/cuda-12.4/targets/ppc64le-linux/lib
   ```
   否则会静默加载 11.8 的库。
2. **服务不能用 sudo 起**（sudo 会清掉 `LD_LIBRARY_PATH`）；但 `pip install` 到 root 属主的 venv 需要 sudo。

---

## 三、目录说明

| 路径 | 内容 |
|---|---|
| `vllm-omni-0.26.0/` | **vLLM-Omni 0.26.0 完整源码**（现役打过补丁的工作树，62MB，2789 文件）。本机不是 git 仓库，是从 tarball 解压的，所以**没有 diff 基线**。 |
| `vllm-omni-0.26.0/vllm_omni/diffusion/models/minimax_h3/` | **H3 移植核心**（含大量 `.bak_*` 历史版本，可看演进）：`pipeline_minimax_h3.py`、`minimax_h3_transformer.py`、`denoise_loop.py`、`encoder.py`、`vae.py`、`packed_sequence.py`、`scheduling_minimax_h3_euler_ancestral.py` |
| `h3_custom/` | **19 个 `h3_*.py` 自定义模块**，AC922 移植特有，装在 venv site-packages 里被 pipeline 直接 import。见下表。 |
| `vllm_core_patch/cuda_view.cu` | 对 vLLM 核心**唯一的源码改动**（torch 2.10/2.11 ABI 适配）。vLLM 源码树本身 3.6GB，只带这一个文件。 |
| `model_config/` | H3 模型**配置 json**（不含权重）：transformer/、text_encoder/、video_vae/、audio_vae/、processor/ 的 config 与 index |
| `repro/` | 复现脚本：`_gen_req.py`（构造 440Hz 参考音频 + 发 multipart 请求）、`h3_ref.png`（参考图，红苹果）、`_env_probe.py` |
| `logs/` | 服务端日志、VAE round-trip 日志 |
| `outputs/` | 实际产出的 MP4 与抽帧（含参考输入图对照） |

### `h3_custom/` 各模块职责

| 文件 | 作用 |
|---|---|
| **`h3_fp16_mixed.py`** | ⭐ **能跑起来的必要条件**。V100 上纯 bf16 DiT 会直接 NaN，这个补丁改残差流精度。禁用它（`H3_NO_FP16_PATCH=1`）会立刻 `RuntimeError: v must be finite`。 |
| **`h3_dit_checkpoint.py`** | DiT 权重加载 + **TP 分片**（按 `get_tensor_model_parallel_world_size()` 切 gate/up 等）。内置 `H3_DEQUANT_MODE=1` 开关。 |
| **`h3_chunked_sdpa.py`** | monkeypatch `SDPAImpl._forward_impl`，V100 分块 query（chunk=128）+ 全 key softmax |
| `h3_vae_proxy.py` / `h3_vae_worker.py` | VAE 代理/offload |
| `h3_stream_encoder.py` | 文本编码器流式加载 |
| `h3_nan_probe.py` | 可选 NaN 定位探针（`H3_FP16_DEBUG`） |
| 其余 `*_remote.py` / `*_probe.py` / `*_test*.py` | 历史一次性诊断脚本（权重离群值、TE dtype/norm 审计、QKV 探针、latent CPU 版等），**不参与生产路径** |

---

## 四、一键复现

### 1) 起服务

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:/usr/local/cuda-12.4/targets/ppc64le-linux/lib
export TORCHDYNAMO_DISABLE=1                 # ⭐ ppc64le 无 Triton，不开必炸
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=3600     # 默认 600s 不够（api_server.py:2955）

setsid bash -c '/srv/omni/venv/bin/vllm-omni serve /srv/models/MiniMax-H3-FP8/Ref2VA \
  --omni --port 8000 --tensor-parallel-size 4 --diffusion-attention-backend TORCH_SDPA \
  > /tmp/h3_sdpa.log 2>&1' >/dev/null 2>&1 </dev/null &
```

> 必须用 `setsid ... &` 的形式，否则 SSH 通道一断进程就死。
> **千万不要在启动脚本里写 `pkill -f 'vllm-omni serve'`** —— 它会匹配到启动脚本自己，把自己杀掉。

### 2) 发请求

```bash
/srv/omni/venv/bin/python /tmp/_gen_req.py     # 即 repro/_gen_req.py
# 成功时输出：DONE 200 <字节数>，结果落在 /tmp/h3_out_a4.bin
```

### 3) 硬性约束（踩过的坑）

- **这个 checkpoint 分区只支持 `ref2va`**，必须给 **参考图 + 参考音频**（或纯视频）。
  传 `task=t2va` 会被拒：`checkpoint partition 'ref2va' supports ['ref2va']`。
- `POST /v1/videos/sync` 是 **multipart**：
  - `input_reference=@xxx.png`（文件上传字段）
  - `audio_reference={"audio_url":"data:audio/wav;base64,..."}`（**JSON 字符串**，支持 data URI）
- **必须显式传 `width` / `height`**（默认 768×1344，V100 扛不住）、`fps=24`（固定值）、`num_frames`（默认 209）。
- `--tensor-parallel-size 4`：TP=1 时 DiT 权重单卡装不下（`h3_dit_checkpoint.py` 会 OOM）。
- **输出封装只走 PyAV**（代码不走 ffmpeg 二进制）。PyAV 需要 `av==11.0.0` + 系统 ffmpeg 5.1 dev 包；
  PyAV 11 没有顶层 `av.VideoStream`，本机在 `media_utils.py` 加了兼容 shim。

---

## 五、诊断开关速查（都在 `denoise_loop.py` / `h3_dit_checkpoint.py` 内置）

| 环境变量 | 作用 |
|---|---|
| `H3_STEP_TRACE=1` | 每步打印 x / v / x0_pred 的 std、corr(x0,x) |
| `H3_SCALE_TRACE=1` | 目标 / 参考幅度比 |
| `H3_SENSITIVITY=1` | (A) 时间步扫描 (B) 清零参考图 (C) 清零文本 (E) AdaLN gate |
| `H3_ATTN_PROBE=1` (+`_EXIT=1`) | 换掉一半 token 看跨 token 信息流；`_EXIT` 表示探针完立刻退出（会返回 500，属预期） |
| `H3_DEQUANT_MODE=1` | 加载时一次反量化成 bf16 常驻 —— **TP=4 下会 OOM，别试** |
| `H3_NO_FP16_PATCH=1` | 禁用 fp16 补丁走纯 bf16 —— **会 NaN，仅供对照** |

---

## 六、想让人帮看什么

**核心问题**：为什么 DiT 出来的 video latent 统计上正常、语义上却是噪声？

三个收敛方向（详见 ANALYSIS.md 末尾）：

1. `MiniMaxH3Transformer3DModel` 前向移植里 **video 特有路径**接线错了
   （`video_patch_proj` / video RoPE / 位置编码 / qk_norm）
2. checkpoint 本身的问题（fork 作者注释里也提到 `near-constant velocity → checkpoint or conditioning`）
3. 需要一个**健康基线**对照 —— 现在无法断定「跨 token 信息流 5~7%」是异常还是 t=0 处本就如此。

> 注：本机 SSH 辅助脚本（`_ssh.py`）因内含凭据，**未** 纳入本仓库。
