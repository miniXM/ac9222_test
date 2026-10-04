# 实际产出对照

| 文件 | 说明 |
|---|---|
| `h3_reference_input.png` | **参考输入图**（红苹果，512×512）—— 用来判断输出是否"语义正确"的基准 |
| `h3_sdpa_256x256.mp4` | **服务实际产出的视频**：637KB，39 帧 256×256 @24fps，含音轨（HTTP 200） |
| `h3_sdpa_frame_first.png`<br>`h3_sdpa_frame_mid.png`<br>`h3_sdpa_frame_last.png` | 输出视频的首/中/末帧 —— **结构化的 16×16 块状马赛克**，块尺寸正好等于 latent 网格 256/16 |
| `h3_vae_rt_fp32_f0.png`<br>`h3_vae_rt_fp32_fmid.png` | **VAE round-trip 重建结果（fp32 解码）**：清晰完整的苹果，无任何块状伪影 → 证明 VAE 健康 |
| `h3_vae_rt_fp16_fmid.png` | 同上但用 fp16 autocast 解码 —— 与 fp32 **逐位一致** |

**核心对照**：把 `h3_reference_input.png` → `h3_vae_rt_fp32_fmid.png`（VAE 自己编解码，正常）
和 `h3_reference_input.png` → `h3_sdpa_frame_mid.png`（完整 pipeline，马赛克）放在一起看，
问题就锁定在 DiT 产出的 latent，而不是 VAE。

## FA-V100 轮（2026-10-05 补）

| 文件 | 说明 |
|---|---|
| `h3_fav100_256x256.mp4` | **真·FA-V100 kernel 跑出的视频**：637,179B，39帧，统计量与 SDPA 版几乎相同（亮度均值 158.03 vs 158） |
| `h3_fav100_frame_first/mid/last.png` | FA-V100 输出帧 —— **与 SDPA 的马赛克形态一致** |

→ 两个完全不同的 attention kernel（torch SDPA vs V100 FA CUDA）产出相同的错误结果，
**attention 实现彻底排除**。详见 `../h3_custom/FA_V100_NOTES.md` 与 `../ANALYSIS.md`。
