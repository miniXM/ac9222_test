"""CPU-only: read the text encoder checkpoint's norm weights and input_scale keys.

Tests the folded-input_scale hypothesis: if ModelOpt folded the per-tensor
activation scale into the preceding RMSNorm weights, the stored norm weights
are NOT ~1.0 but ~input_scale (possibly tiny/large). Running such a checkpoint
through plain bf16/fp16 GEMMs without un-folding gives activations at the
wrong scale, which poisons every downstream layer.
"""
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path("/srv/models/MiniMax-H3-FP8/Ref2VA/text_encoder")
idx = json.loads((ROOT / "model.safetensors.index.json").read_text())
wm = idx["weight_map"]
handles = {}


def get(k):
    f = wm[k]
    if f not in handles:
        handles[f] = safe_open(str(ROOT / f), framework="pt", device="cpu")
    return handles[f].get_tensor(k)


keys = set(wm.keys())
n_is = sum(1 for k in keys if k.endswith(".input_scale"))
n_ws = sum(1 for k in keys if k.endswith(".weight_scale"))
n_fp8 = 0
for k in keys:
    if k.endswith(".weight"):
        try:
            if get(k).dtype == torch.float8_e4m3fn:
                n_fp8 += 1
        except Exception:  # noqa: BLE001
            pass

out = {
    "total_keys": len(keys),
    "input_scale_keys": n_is,
    "weight_scale_keys": n_ws,
    "fp8_weights": n_fp8,
    "norms": {},
    "sample_input_scales": {},
}

# norm weights: layer0 norms + final norm + embed
for k in sorted(keys):
    if ("input_layernorm" in k or "post_attention_layernorm" in k
            or k.endswith("model.language_model.norm.weight")
            or "q_norm.weight" in k or "k_norm.weight" in k):
        if "layers.0." in k or k.endswith("model.language_model.norm.weight"):
            t = get(k).float()
            out["norms"][k] = {
                "dtype": str(get(k).dtype),
                "shape": list(t.shape),
                "absmax": round(float(t.abs().max()), 6),
                "std": round(float(t.std()), 6),
                "mean": round(float(t.mean()), 6),
            }

# a few input_scales with their associated weights
for k in sorted(keys):
    if k.endswith(".input_scale") and len(out["sample_input_scales"]) < 12:
        t = get(k).float()
        out["sample_input_scales"][k] = {
            "shape": list(t.shape),
            "value": round(float(t.reshape(-1)[0]), 6) if t.numel() else None,
            "absmax": round(float(t.abs().max()), 6),
        }

print(json.dumps(out, indent=2))
with open("/tmp/te_norm_audit.json", "w") as f:
    json.dump(out, f, indent=2)
