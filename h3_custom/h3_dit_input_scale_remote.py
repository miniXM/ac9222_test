"""CPU-only: compare the scale of the three token groups entering the DiT.

The DiT consumes tokens in one packed sequence:
  text rows   = context_embedder(text_hidden)          [5120 -> hidden]
  cond rows   = x_embedder(reference latent tokens)    [96   -> hidden]
  target rows = x_embedder(noisy latent tokens)        [96   -> hidden]
If the text branch enters at a wildly different scale than the video branch,
attention to text tokens dominates every block and the model cannot denoise.
"""
import json
import os
import re
from pathlib import Path

import torch
from safetensors import safe_open

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
DIT = Path("/srv/models/MiniMax-H3-FP8/Ref2VA/transformer")
idx = json.loads((DIT / "diffusion_pytorch_model.safetensors.index.json").read_text())
wm = idx["weight_map"]
handles = {}


def get(k):
    f = wm[k]
    if f not in handles:
        handles[f] = safe_open(str(DIT / f), framework="pt", device="cpu")
    return handles[f].get_tensor(k)


def dequant(k):
    t = get(k)
    if t.dtype == torch.float8_e4m3fn:
        s = get(k[: -len("weight")] + "weight_scale").float()
        return t.float() * s.reshape(-1, *([1] * (t.dim() - 1)))
    return t.float()


out = {}

# --- weights of the two entry projections
ctx_w = dequant("context_embedder.weight")
ctx_b = dequant("context_embedder.bias") if "context_embedder.bias" in wm else None
xe_w = dequant("proj_in.weight")
out["x_embedder_key"] = "proj_in.weight"
out["ctx_w_std"] = round(float(ctx_w.std()), 6)
out["ctx_w_shape"] = list(ctx_w.shape)
out["xe_w_std"] = round(float(xe_w.std()), 6)
out["xe_w_shape"] = list(xe_w.shape)

# --- video / cond token branch scale (from the saved final latent)
d = torch.load("/tmp/h3_final_latent.pt", map_location="cpu", weights_only=False)
rows = d["video_rows"].float()
um = d["update_mask"]
video_in = xe_w @ rows.T  # [hidden, tokens]
out["video_target_in_std"] = round(float(video_in[:, um].std()), 4)
out["video_cond_in_std"] = round(float(video_in[:, ~um].std()), 4)

# --- text token branch scale, for both dtype variants of the encoder
for tag in ("fp16", "bf16"):
    p = Path(f"/tmp/h3_te_dtype_test/hidden_{tag}.pt")
    if not p.exists():
        continue
    h = torch.load(p, map_location="cpu", weights_only=True).float()
    if h.dim() == 3:
        h = h[0]
    text_in = ctx_w @ h.T
    if ctx_b is not None:
        text_in = text_in + ctx_b[:, None]
    out[f"text_in_std_{tag}"] = round(float(text_in.std()), 4)
    out[f"text_in_absmax_{tag}"] = round(float(text_in.abs().max()), 4)
    out[f"text_hidden_std_{tag}"] = round(float(h.std()), 4)
    out[f"text_hidden_absmax_{tag}"] = round(float(h.abs().max()), 4)

# ratio text/video entering the DiT
if "text_in_std_fp16" in out:
    out["ratio_text_vs_video_fp16"] = round(
        out["text_in_std_fp16"] / max(out["video_target_in_std"], 1e-9), 2
    )
if "text_in_std_bf16" in out:
    out["ratio_text_vs_video_bf16"] = round(
        out["text_in_std_bf16"] / max(out["video_target_in_std"], 1e-9), 2
    )

print(json.dumps(out, indent=2))
with open("/tmp/h3_dit_input_scale.json", "w") as f:
    json.dump(out, f, indent=2)
