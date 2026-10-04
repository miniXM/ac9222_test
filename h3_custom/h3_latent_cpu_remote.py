"""Remote CPU-only analysis of the latent produced by the last H3 sampling run.

No CUDA device is visible to this process.
Answers:
  1. Are the conditioning rows (reference image) degenerate (zero/constant)?
  2. Does the denoised target latent already carry an alternating component?
  3. Is the target latent still essentially unstructured?
"""
import json
import os
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

d = torch.load("/tmp/h3_final_latent.pt", map_location="cpu", weights_only=False)
rows = d["video_rows"].float()
um = d["update_mask"]
lt, lh, lw = d["latent_t"], d["latent_h"], d["latent_w"]
out = {"keys": sorted(d.keys()), "video_rows": list(rows.shape),
       "latent_t_h_w": [int(lt), int(lh), int(lw)]}


def bs(name, t):
    t = t.float()
    if t.numel() == 0:
        return {"name": name, "empty": True}
    f = t.reshape(t.shape[0], -1)
    return {
        "name": name,
        "shape": list(t.shape),
        "std": round(float(t.std()), 5),
        "mean": round(float(t.mean()), 5),
        "absmax": round(float(t.abs().max()), 5),
        "frac_exact_zero": round(float((t == 0).float().mean()), 5),
        "row_std_mean": round(float(f.std(dim=1).mean()), 5),
        "row_std_min": round(float(f.std(dim=1).min()), 5),
    }


cond = rows[~um] if int((~um).sum()) else rows[:0]
tgt = rows[um]
out["cond_rows_n"] = int((~um).sum())
out["target_rows_n"] = int(um.sum())
out["cond_video_rows"] = bs("cond_video_rows", cond)
out["target_video_rows"] = bs("target_video_rows", tgt)

# checkerboard signature inside the denoised target latent
th, tw = int(lh) // 2, int(lw) // 2
C = tgt.shape[-1]
try:
    g = tgt.reshape(int(lt), th, tw, C)
    def lag(a, b):
        return (((a - a.mean()) * (b - b.mean())).mean()
                / (a.std() * b.std() + 1e-9)).item()
    out["latent_lag_h"] = round(lag(g[:, : th - 1], g[:, 1:]), 4)
    out["latent_lag_w"] = round(lag(g[:, :, : tw - 1], g[:, :, 1:]), 4)
    out["latent_grid"] = [int(lt), th, tw, C]
except Exception as e:  # noqa: BLE001
    out["grid_error"] = repr(e)[:200]

print(json.dumps(out, indent=2))
with open("/tmp/h3_vae_decode_test/latent_cpu.json", "w") as f:
    json.dump(out, f, indent=2)
