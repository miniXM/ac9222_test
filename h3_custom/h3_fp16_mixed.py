"""Experiment-only FP16 mixed precision for MiniMax H3 on V100 (sm_70).

Source of the approach (all three agree, verified on real V100 renders):
  * ComfyUI issue #15262 -- H3 residual stream reaches ~1e6..1e7, fp16 tops out
    at 65504, so naive fp16 compute is structurally impossible.
  * github.com/Amduraznak/minimax-h3-fp16-fix -- keep residual fp32, cast the
    normalized/modulated branch inputs to fp16 so attn/MLP matmuls still hit
    fp16 tensor cores, rescale out_proj/fc2 by a power of two (bit-exact).
    Measured 30 s/step vs 330-370 s/step fp32 (~11x) on a V100.
  * ComfyUI-MiniMaxH3-FP16Safe v6.8.0 -- additionally: RMSNorm in fp32,
    always-fp16 SDPA with a fixed /16 power-of-two input scale (q/k restored
    by RMSNorm homogeneity, v unscaled after out_proj), MLP fully fp16 with
    gate branch /16 and fc2 input /8.

Why this matters on V100: there is no bf16 tensor core. bf16 matmul runs on
the CUDA cores (~7.9 TFLOPS measured here) while fp16 uses the tensor cores
(~125 TFLOPS peak) -- and cuBLAS fp16 GEMM accumulates in fp32 by default,
which is exactly the "fp16 compute, fp32 accumulate" the user asked for.

What this module changes relative to the stock bf16 path:
  1. residual stream: fp32 across all 50 DiT blocks and the 2 refiner blocks
  2. attention: input x/16 (power of two) -> fp16 qkv, q/k RMSNorm restores
     O(1) magnitude, v stays /16 through out_proj, unscaled x16 in fp32
  3. MLP: fully fp16, silu(gate) * (up/16), fc2 input /8, output x128 in fp32
  4. RMSNorm: fp32 compute (weight upcast to fp32), I/O dtype preserved
  5. condition_proj stays bf16 -- its output reaches ~96k, beyond fp16
  6. final output heads / patch projections / time embedder stay fp32
"""
import os
import time

import torch
import torch.nn.functional as F

from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as M

FP16 = torch.float16
BF16 = torch.bfloat16
FP32 = torch.float32

# ``_run_packed_attention`` reads two cu_seqlens entries with .item() on every
# call: that is a GPU->CPU sync per attention, i.e. ~100 per step, and each one
# has to drain the queue (and the pending TP collectives) before it returns.
# The tensor is fixed for the whole DiT forward, so cache it and flush per call.
_CU_CACHE = {}

# Power-of-two scales: exact in fp16 (only the exponent moves).
ATTN_SCALE = 16.0   # 2^4; FP16Safe measured /16 keeps out_proj output <= ~2400
GATE_SCALE = 16.0   # fc1 output max ~585 -> gated product ~21.4k (fp16-safe)
FC2_SCALE = 8.0     # fc2 output ~8.5x input -> <= ~22.7k

DEBUG = os.environ.get("H3_FP16_DEBUG", "0") == "1"


def _amax(tag, t):
    if not DEBUG:
        return
    try:
        v = t.detach().abs().max().item()
        f = bool(torch.isfinite(t.detach()).all().item())
        print(f"H3_FP16_TRACE {tag} amax={v:.4g} finite={f}", flush=True)
    except Exception as ex:
        print(f"H3_FP16_TRACE {tag} probe failed: {ex}", flush=True)


def _patch_norm(module):
    """RMSNorm: compute in fp32, return the caller's dtype (no sync)."""
    if getattr(module, "_h3_fp32_norm", False):
        return
    shape = module.normalized_shape
    eps = module.eps
    module.weight = torch.nn.Parameter(module.weight.detach().to(FP32), requires_grad=False)
    module._h3_fp32_norm = True

    def forward(x, _s=shape, _e=eps, _m=module):
        # F.rms_norm(input, normalized_shape, weight, eps) -- no bias arg here.
        return F.rms_norm(x.float(), _s, _m.weight, _e).to(x.dtype)

    module.forward = forward


def _run_packed_attention(self, q, k, v, *, cu_seqlens, max_seqlen):
    """Same as stock, but takes the two scalar bounds from the cache."""
    bounds = getattr(self, "_h3_cu", None)
    if bounds is None:
        bounds = (int(cu_seqlens[1].item()), int(cu_seqlens[-1].item()))
        self._h3_cu = bounds
    used, packed_total = bounds
    attn_mask = None
    if used < packed_total:
        attn_mask = torch.arange(packed_total, device=q.device)[None] < used
    if os.environ.get("H3_ATTN_UNIT", "0") == "1" and not getattr(self, "_h3_unit", False):
        try:
            self._h3_unit = True
            with torch.no_grad():
                _m = M.AttentionMetadata(
                    attn_mask=attn_mask,
                    extra={"cu_seqlens_q": cu_seqlens, "cu_seqlens_k": cu_seqlens,
                           "max_seqlen_q": max_seqlen, "max_seqlen_k": max_seqlen})
                _o1 = self.attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), _m).squeeze(0)
                _ti = min(5, q.shape[0] - 1)
                _k2 = k.clone(); _k2[_ti] = _k2[_ti] + 10.0
                _v2 = v.clone(); _v2[_ti] = _v2[_ti] + 10.0
                _o2 = self.attention(q.unsqueeze(0), _k2.unsqueeze(0), _v2.unsqueeze(0), _m).squeeze(0)
            _d = (_o2.float() - _o1.float())
            _d_tok = _d.reshape(_d.shape[0], -1).norm(dim=-1)
            _b_tok = _o1.float().reshape(_o1.shape[0], -1).norm(dim=-1).clamp_min(1e-9)
            _rel = _d_tok / _b_tok
            _others = _rel.clone(); _others[_ti] = 0.0
            with open("/tmp/h3_attn_probe.log", "a") as _f:
                _f.write("H3_ATTN_UNIT T=%d changed=%d rel[changed]=%.6f "
                         "rel_others_mean=%.8f rel_others_max=%.8f n_affected=%d\n"
                         % (_rel.shape[0], _ti, float(_rel[_ti]),
                            float(_others.mean()), float(_others.max()),
                            int((_others > 1e-3).sum())))
        except Exception as _ex:
            try:
                with open("/tmp/h3_attn_probe.log", "a") as _f:
                    import traceback as _tb
                    _f.write("H3_ATTN_UNIT failed: %r\n" % (_ex,))
                    _tb.print_exc(file=_f)
            except Exception:
                pass
    metadata = M.AttentionMetadata(
        attn_mask=attn_mask,
        extra={
            "cu_seqlens_q": cu_seqlens,
            "cu_seqlens_k": cu_seqlens,
            "max_seqlen_q": max_seqlen,
            "max_seqlen_k": max_seqlen,
        },
    )
    return self.attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), metadata).squeeze(0)


def _attn_forward(self, x, *, rope_freqs, cu_seqlens, max_seqlen, sp_seq_lens=None):
    key = (cu_seqlens.data_ptr(), cu_seqlens.shape, cu_seqlens.device)
    bounds = _CU_CACHE.get(key)
    if bounds is None:
        bounds = (int(cu_seqlens[1].item()), int(cu_seqlens[-1].item()))
        _CU_CACHE[key] = bounds
    self._h3_cu = bounds
    total = x.shape[0]
    xh = (x.float() * (1.0 / ATTN_SCALE)).to(FP16)
    qkv, _ = self.qkv_proj(xh)
    q_size = self.num_heads * self.head_dim
    kv_size = self.num_kv_heads * self.head_dim
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    q = q.view(total, self.num_heads, self.head_dim)
    k = k.view(total, self.num_kv_heads, self.head_dim)
    v = v.view(total, self.num_kv_heads, self.head_dim)
    # RMSNorm is scale-homogeneous, so q/k come back to O(1) after the /16.
    q = self.q_norm(q)
    k = self.k_norm(k)
    if rope_freqs is not None:
        q = M._apply_rope(q, rope_freqs).to(FP16)
        k = M._apply_rope(k, rope_freqs).to(FP16)
    out = self._run_packed_attention(q, k, v, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
    out = out.reshape(total, self.num_heads * self.head_dim)
    out, _ = self.out_proj(out)
    # v was never rescaled; undo the /16 in fp32.
    return out.float() * ATTN_SCALE


def _mlp_forward(self, x):
    xh = x.to(FP16)
    hidden, _ = self.fc1(xh)
    gate, up = hidden.chunk(2, dim=-1)
    act = F.silu(gate) * (up * (1.0 / GATE_SCALE))
    out, _ = self.fc2(act * (1.0 / FC2_SCALE))
    return out.float() * (FC2_SCALE * GATE_SCALE)


def _block_forward(self, x, *, t_emb, combined_indices, rope_freqs, cu_seqlens,
                   max_seqlen, sp_seq_lens=None):
    (shift_msa, scale_msa, gate_msa,
     shift_mlp, scale_mlp, gate_mlp) = self.adaln_proj(t_emb)
    x = x.float()
    residual = x
    h = self.norm1(x)
    h = M._modulate_scale_shift(h, shift_msa, scale_msa, combined_indices, dtype=FP16)
    h = self.attn(h, rope_freqs=rope_freqs, cu_seqlens=cu_seqlens,
                  max_seqlen=max_seqlen, sp_seq_lens=sp_seq_lens)
    x = M._modulate_gate(residual, gate_msa, h, combined_indices, dtype=FP32)
    residual = x
    h = self.norm2(x)
    h = M._modulate_scale_shift(h, shift_mlp, scale_mlp, combined_indices, dtype=FP16)
    h = self.mlp(h)
    out = M._modulate_gate(residual, gate_mlp, h, combined_indices, dtype=FP32)
    _amax(f"block{getattr(self, '_dbg_index', -1)}", out)
    return out


def _refiner_block_forward(self, x, *, cu_seqlens, max_seqlen):
    x = x.float()
    h = self.norm1(x).to(FP16)
    x = x + self.attn(h, rope_freqs=None, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
    h = self.norm2(x).to(FP16)
    return x + self.mlp(h)


def _final_forward(self, x, *, t_emb, inverse_indices):
    shift, scale = self.adaln_proj(t_emb)
    h = self.norm(x.float())
    _tr = os.environ.get("H3_FINAL_TRACE", "0") == "1"
    if _tr:
        print("H3_FINAL x_in    std=%.5g amax=%.5g"
              % (float(x.detach().float().std()), float(x.detach().float().abs().max())), flush=True)
        print("H3_FINAL h_norm  std=%.5g amax=%.5g"
              % (float(h.detach().float().std()), float(h.detach().float().abs().max())), flush=True)
        _sh = shift.detach().float().reshape(-1)
        _sc = scale.detach().float().reshape(-1)
        print("H3_FINAL shift   std=%.5g mean=%.5g | scale std=%.5g mean=%.5g  (1+scale) mean=%.5g"
              % (float(_sh.std()), float(_sh.mean()), float(_sc.std()),
                 float(_sc.mean()), float(1.0 + _sc.mean())), flush=True)
    h = M._modulate_scale_shift(h, shift, scale, inverse_indices, dtype=FP32)
    h = h.to(FP32)
    if _tr:
        print("H3_FINAL h_mod   std=%.5g amax=%.5g"
              % (float(h.detach().std()), float(h.detach().abs().max())), flush=True)
        _w = self.video_out.weight.detach().float()
        print("H3_FINAL w_out   std=%.5g absmax=%.5g shape=%s"
              % (float(_w.std()), float(_w.abs().max()), tuple(_w.shape)), flush=True)
    video, _ = self.video_out(h)
    audio, _ = self.audio_out(h)
    if _tr:
        _v = video.detach().float()
        _u = _v[_v.abs() < 1e6]
        print("H3_FINAL v_out   std=%.5g amax=%.5g  | finite_std=%.5g"
              % (float(_v.std()), float(_v.abs().max()),
                 float(_u.std()) if _u.numel() else float("nan")), flush=True)
    return video, audio


def _make_embed():
    def _embed(self, *, x, audio_x, text_embeddings_selected, unique_timesteps,
               img_pos, audio_pos, text_pos, refiner_cu_seqlens,
               refiner_max_seqlen, seq_len, device):
        x_rows = x.view(-1, x.shape[-1]).index_select(0, img_pos).to(FP32)
        video_embed, _ = self.video_patch_proj(x_rows)
        audio_rows = audio_x.view(-1, audio_x.shape[-1]).index_select(0, audio_pos).to(FP32)
        audio_embed, _ = self.audio_patch_proj(audio_rows)

        # condition_proj output reaches ~96k: keep it on the bf16 path.
        text_rows = text_embeddings_selected.to(device=device, dtype=BF16)
        text_embed, _ = self.condition_proj(text_rows)
        text_embed = self.token_refiner(
            text_embed.float(),
            cu_seqlens=refiner_cu_seqlens,
            max_seqlen=refiner_max_seqlen,
        )

        # fp32 accumulation buffer: the residual stream must never be fp16.
        embeddings = torch.zeros((seq_len, self.hidden_size), device=device, dtype=FP32)
        embeddings.index_add_(0, text_pos, text_embed.to(FP32)[: text_pos.shape[0]])
        embeddings.index_add_(0, img_pos, video_embed.to(FP32)[: img_pos.shape[0]])
        embeddings.index_add_(0, audio_pos, audio_embed.to(FP32)[: audio_pos.shape[0]])
        t_emb = self.time_embedder(unique_timesteps)

        if os.environ.get("H3_EMBED_TRACE", "0") == "1":
            def _st(t):
                t = t.detach()
                return (float(t.abs().max()), float(t.float().std()))

            ta, ts = _st(text_rows)
            ea, es = _st(text_embed)
            va, vs = _st(video_embed)
            aa, as_ = _st(audio_embed)
            print("H3_EMBED text_rows   n=%d absmax=%.5g std=%.5g" % (text_rows.shape[0], ta, ts), flush=True)
            print("H3_EMBED text_embed  n=%d absmax=%.5g std=%.5g" % (text_embed.shape[0], ea, es), flush=True)
            print("H3_EMBED video_embed n=%d absmax=%.5g std=%.5g" % (video_embed.shape[0], va, vs), flush=True)
            print("H3_EMBED audio_embed n=%d absmax=%.5g std=%.5g" % (audio_embed.shape[0], aa, as_), flush=True)
            _tp = text_pos.view(-1).to(torch.long)
            _ip = img_pos.view(-1).to(torch.long)
            _ap = audio_pos.view(-1).to(torch.long)
            for nm, pp in (("text", _tp), ("img", _ip), ("audio", _ap)):
                if pp.numel():
                    g = embeddings.index_select(0, pp)
                    ga, gs = _st(g)
                    print("H3_EMBED emb[%s] n=%d absmax=%.5g std=%.5g" % (nm, pp.numel(), ga, gs), flush=True)
            print("H3_EMBED seq_len=%d sum_rows=%d" % (seq_len, _tp.numel() + _ip.numel() + _ap.numel()), flush=True)

        return embeddings, t_emb
    return _embed


def install(model):
    """Patch the loaded DiT model in place. Call after weights are loaded."""
    # 1. class-level forward replacements
    M.MiniMaxH3Attention.forward = _attn_forward
    M.MiniMaxH3Attention._run_packed_attention = _run_packed_attention
    M.MiniMaxH3MLP.forward = _mlp_forward
    M.MiniMaxH3DiTBlock.forward = _block_forward
    M.MiniMaxH3TokenRefinerBlock.forward = _refiner_block_forward
    M.MiniMaxH3FinalLayer.forward = _final_forward

    # Opt-in numerical control: scaling q/k by 1/S requires eps/S^2.
    # Leave the baseline unchanged unless explicitly requested by the probe.
    if os.environ.get("H3_QK_EPS_FIX", "0") == "1":
        for m in model.modules():
            if isinstance(m, M.MiniMaxH3Attention):
                for norm in (m.q_norm, m.k_norm):
                    if not getattr(norm, "_h3_eps_compensated", False):
                        norm.eps /= ATTN_SCALE ** 2
                        norm._h3_eps_compensated = True
        print("H3_QK_EPS_FIX enabled", flush=True)

    # 2. fp32 RMSNorm everywhere in the DiT
    norms = 0
    for m in model.modules():
        if isinstance(m, torch.nn.RMSNorm):
            _patch_norm(m)
            norms += 1

    # 3. per-instance embedding path (fp32 scatter buffer)
    model._embed = _make_embed().__get__(model, type(model))

    # 4. cheap per-DiT-call timing: no sync, just wall time between submits.
    #    Once the GPU queue is full the CPU blocks, so this tracks step cost.
    original_model_forward = M.MiniMaxH3DiTModel.forward
    stat = {"n": 0}

    def model_forward(self, **kw):
        # cu_seqlens tensors are reused across steps with different contents.
        _CU_CACHE.clear()
        t = time.time()
        out = original_model_forward(self, **kw)
        stat["n"] += 1
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"H3_DIT_CALL {stat['n']} {time.time() - t:.2f}s "
              f"peak={peak:.2f}GiB", flush=True)
        return out

    if not getattr(M.MiniMaxH3DiTModel, "_h3_fp16_timed", False):
        M.MiniMaxH3DiTModel.forward = model_forward
        M.MiniMaxH3DiTModel._h3_fp16_timed = True
    for name, mod in model.named_modules():
        if isinstance(mod, M.MiniMaxH3TokenRefinerBlock):
            mod._dbg_index = -1
    for i, blk in enumerate(getattr(model, "blocks", [])):
        blk._dbg_index = i

    print(f"H3_FP16_MIXED installed: norms_fp32={norms} attn_scale={ATTN_SCALE} "
          f"gate={GATE_SCALE} fc2={FC2_SCALE} debug={DEBUG}", flush=True)
    return model
