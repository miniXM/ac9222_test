"""Remote-side decisive VAE decoder test.

Question under test: is the checkerboard produced by the VAE decoder itself
(precision / weights), or is it already present in the latent the DiT produced?

Method: decode three latents under three precisions.
  real   = the latent actually produced by the last sampling run
  zeros  = all-zero latent (must decode to a flat, structureless frame)
  smooth = low-frequency latent (must decode to a smooth gradient)
If `zeros` / `smooth` come out as checkerboard under fp16 but clean under
fp32/bf16, the decoder precision is the fault. If they come out clean under
every precision, the decoder is fine and the fault is upstream in the DiT.
"""
import os
import json
import torch

torch.backends.cudnn.enabled = False  # POWER9 GET engine failure fallback

from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE
from vllm_omni.diffusion.models.minimax_h3.packed_tokens import (
    minimax_h3_unpatchify_video_tokens,
)

OUT = "/tmp/h3_vae_decode_test"
os.makedirs(OUT, exist_ok=True)
report = []


def stats(name, t):
    t = t.detach().float()
    finite = torch.isfinite(t).all().item()
    # lag-1 spatial correlation along H and W separately. A 2-period checkerboard
    # gives strongly NEGATIVE single-axis correlation (diagonal shift would be
    # positive and must not be used).
    h, w = t.shape[-2], t.shape[-1]

    def axis_corr(a, b):
        if a.numel() > 1 and a.std() > 0 and b.std() > 0:
            return float((((a - a.mean()) * (b - b.mean())).mean()
                          / (a.std() * b.std())).item())
        return float("nan")

    corr_h = axis_corr(t[..., : h - 1, :], t[..., 1:, :])
    corr_w = axis_corr(t[..., :, : w - 1], t[..., :, 1:])
    return {
        "name": name,
        "shape": list(t.shape),
        "finite": bool(finite),
        "std": round(float(t.std()), 5),
        "min": round(float(t.min()), 4),
        "max": round(float(t.max()), 4),
        "lag1_corr_h": round(corr_h, 4),
        "lag1_corr_w": round(corr_w, 4),
    }


def save_frame(name, video):
    """Save the first frame. Decoded layout is (B,C,T,H,W)."""
    import numpy as np
    from PIL import Image

    v = video.detach().float().cpu()
    print("SAVE_SHAPE", name, list(v.shape), flush=True)
    if v.dim() == 5:  # (B,C,T,H,W)
        v = v[0, :, 0]  # (C,H,W) first frame
    elif v.dim() == 4:  # (B,T,H,W,C) fallback
        v = v[0, 0]
        if v.shape[0] in (3, 4) and v.shape[-1] not in (3, 4):
            v = v.permute(2, 0, 1)
    while v.dim() > 3:
        v = v[0]
    if v.shape[0] in (1, 3, 4) and v.shape[-1] not in (1, 3, 4):
        v = v.permute(1, 2, 0)
    if v.shape[-1] == 1:
        v = v[..., 0]
    else:
        v = v[..., :3]
    v = v - v.min()
    m = v.max()
    if m > 0:
        v = v / m
    arr = (v.clamp(0, 1).numpy() * 255).astype("uint8")
    Image.fromarray(arr).save(os.path.join(OUT, name + ".png"))
    return list(arr.shape)


def main():
    d = torch.load("/tmp/h3_final_latent.pt", map_location="cpu", weights_only=False)
    rows = d["video_rows"]
    um = d["update_mask"]
    lt, lh, lw = d["latent_t"], d["latent_h"], d["latent_w"]
    target = rows[um]
    real = minimax_h3_unpatchify_video_tokens(
        target,
        latent_shape=(lt, lh // 2, lw // 2, 24),
        patch_size=(1, 2, 2),
    )
    print("REAL_LATENT", tuple(real.shape), "std", float(real.float().std()), flush=True)

    # Smooth low-frequency latent with matching scale.
    # real shape is (1, C, T, H, W) as produced by the unpatchify helper.
    _, C, T, H, W = real.shape
    tt = torch.linspace(0, 2 * 3.14159, T)[None, None, :, None, None]
    yy = torch.linspace(0, 2 * 3.14159, H)[None, None, None, :, None]
    xx = torch.linspace(0, 2 * 3.14159, W)[None, None, None, None, :]
    cc = torch.linspace(0, 2 * 3.14159, C)[None, :, None, None, None]
    smooth = (torch.sin(tt + yy + xx + cc).expand(1, C, T, H, W).contiguous()
              * float(real.float().std()))

    latents = {
        "real": real,
        "zeros": torch.zeros_like(real),
        "smooth": smooth,
    }

    model = MiniMaxH3VideoVAE(
        "/srv/models/MiniMax-H3-FP8/Ref2VA/video_vae", device=torch.device("cuda:0")
    )

    modes = [
        ("fp16", torch.float16, True),
        ("bf16", torch.bfloat16, True),
        ("fp32", torch.float32, False),
    ]
    for lname, lat in latents.items():
        for mname, dt, use_ac in modes:
            tag = f"{lname}_{mname}"
            # fp16 is the production path and fits the full clip. bf16/fp32
            # exceed 16 GiB for the whole clip, so decode one frame there --
            # the checkerboard is a spatial artefact, one frame is enough.
            cur = lat if mname == "fp16" else lat[..., :1, :, :].contiguous()
            try:
                with torch.inference_mode(), torch.autocast(
                    "cuda", dtype=dt, enabled=use_ac
                ):
                    v = model.decode_latent(cur.to("cuda:0"))
                v = v.detach().float().cpu()
                rec = stats(tag, v)
                rec["decoded_shape"] = list(v.shape)
                print("RESULT", json.dumps(rec), flush=True)
                report.append(rec)
                try:
                    rec["png_shape"] = save_frame(tag, v)
                except Exception as pe:  # noqa: BLE001
                    rec["png_error"] = repr(pe)[:200]
                    print("PNG_ERROR", tag, repr(pe)[:200], flush=True)
            except Exception as e:  # noqa: BLE001
                rec = {"name": tag, "error": repr(e)[:300]}
                print("RESULT", json.dumps(rec), flush=True)
                report.append(rec)
            torch.cuda.empty_cache()

    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("VAE_TEST_DONE", flush=True)


if __name__ == "__main__":
    main()
