"""CPU-only scan of every FP8 weight in the DiT checkpoint.

Motivation: the per-layer activation trace shows hidden_absmax jumping from
171 at layer 5 to 1.51e4 at layer 6 and then staying at ~1.55e4. A single
block that is ~2 orders of magnitude off would explain that jump. This scan
dequantizes every FP8 tensor and reports per-block magnitude statistics so an
outlier block can be identified without running the model.
"""
import json
import os
import re
from pathlib import Path

import torch
from safetensors import safe_open

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path("/srv/models/MiniMax-H3-FP8/Ref2VA/transformer")

idx = json.loads((ROOT / "diffusion_pytorch_model.safetensors.index.json").read_text())
files = sorted(set(idx["weight_map"].values()))
handles = {f: safe_open(str(ROOT / f), framework="pt", device="cpu") for f in files}
keys = set(idx["weight_map"].keys())

rows = []
for k in sorted(keys):
    if not k.endswith(".weight"):
        continue
    t = handles[idx["weight_map"][k]].get_tensor(k)
    if t.dtype != torch.float8_e4m3fn:
        continue
    sk = k[: -len(".weight")] + ".weight_scale"
    if sk not in keys:
        rows.append({"key": k, "error": "missing weight_scale"})
        continue
    s = handles[idx["weight_map"][sk]].get_tensor(sk).float().view(-1)
    w = t.float()
    # per-channel dequantization matching the runtime path
    if s.numel() != w.shape[0]:
        rows.append({"key": k, "error": f"scale rows {s.numel()} != out {w.shape[0]}"})
        continue
    wd = w * s.view(-1, *([1] * (w.dim() - 1)))
    q = w  # raw fp8 value magnitude
    rows.append({
        "key": k,
        "shape": list(w.shape),
        "fp8_absmax": round(float(q.abs().max()), 4),
        "fp8_std": round(float(q.std()), 5),
        "scale_mean": round(float(s.mean()), 6),
        "scale_min": round(float(s.min()), 8),
        "scale_max": round(float(s.max()), 8),
        "deq_std": round(float(wd.std()), 6),
        "deq_absmax": round(float(wd.abs().max()), 5),
    })

print("scanned", len(rows), "fp8 weights")


def block_of(key):
    m = re.search(r"transformer_blocks\.(\d+)", key)
    if m:
        return "block" + m.group(1)
    m = re.search(r"token_refiner|refiner", key)
    if m:
        return "refiner"
    for tag in ("proj_in", "proj_out", "time_embedder", "context_embedder",
                "audio", "image"):
        if tag in key:
            return tag
    return "other"


groups = {}
for r in rows:
    if "error" in r:
        continue
    groups.setdefault(block_of(r["key"]), []).append(r)

summary = {}
for g, items in sorted(groups.items()):
    ds = [i["deq_std"] for i in items]
    da = [i["deq_absmax"] for i in items]
    sm = [i["scale_mean"] for i in items]
    summary[g] = {
        "n": len(items),
        "deq_std_median": round(sorted(ds)[len(ds) // 2], 6),
        "deq_std_max": round(max(ds), 6),
        "deq_absmax_max": round(max(da), 4),
        "scale_mean_median": round(sorted(sm)[len(sm) // 2], 7),
        "scale_mean_max": round(max(sm), 7),
    }

out = {"per_block": summary, "errors": [r for r in rows if "error" in r][:40]}

# Flag any single weight whose dequantized scale is far from its block median.
flags = []
for g, items in groups.items():
    med = summary[g]["scale_mean_median"]
    if med <= 0:
        continue
    for i in items:
        if i["scale_mean"] > med * 20 or i["scale_mean"] < med / 20:
            flags.append({"block": g, "key": i["key"],
                          "scale_mean": i["scale_mean"], "block_median": med,
                          "ratio": round(i["scale_mean"] / med, 2)})
out["scale_outliers"] = sorted(flags, key=lambda x: -abs(x["ratio"]))[:40]

print(json.dumps(out["per_block"], indent=2))
print("SCALE_OUTLIERS", json.dumps(out["scale_outliers"], indent=2))
with open("/tmp/h3_weight_outlier.json", "w") as f:
    json.dump(out, f, indent=2)
