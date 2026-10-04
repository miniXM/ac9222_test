"""FA2-compatible shim backed by flash_attn_v100 (SM70 / V100).

Why this exists
---------------
`flash_attn_v100` 1.2.0 ships a real ppc64le SM70 flash-attention kernel, but
- it only exposes the *dense* `flash_attn_func` (no varlen kernel), and
- it only accepts fp16.

vLLM-Omni's diffusion path needs `flash_attn_varlen_func` (H3 uses packed
sequences with cu_seqlens).  This module synthesises varlen by running the dense
V100 kernel once per packed segment, and casts bf16<->fp16 around the call.
"""
from __future__ import annotations

import torch

__version__ = "2.6.0"          # must satisfy vllm-omni's >= 2.6.0 check

from flash_attn_v100 import flash_attn_func as _v100_func  # noqa: E402

_MARK = "/tmp/h3_fa_v100_used.txt"
_marked = False


def _mark(kind, dtype, shape):
    global _marked
    if _marked:
        return
    _marked = True
    try:
        with open(_MARK, "a") as f:
            f.write("USED %s dtype=%s shape=%s\n" % (kind, dtype, tuple(shape)))
    except Exception:
        pass


def _cast(x):
    return x if x.dtype == torch.float16 else x.to(torch.float16)


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                    window_size=(-1, -1), softcap=0.0, alibi_slopes=None,
                    deterministic=False, return_attn_probs=False, out=None):
    odtype = q.dtype
    _mark("dense", odtype, q.shape)
    o = _v100_func(_cast(q), _cast(k), _cast(v),
                   dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal)
    if isinstance(o, tuple):
        o = o[0]
    return o.to(odtype)


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k=None,
                           max_seqlen_q=None, max_seqlen_k=None,
                           dropout_p=0.0, softmax_scale=None, causal=False,
                           **kwargs):
    """Varlen on top of the dense V100 kernel: one dense call per packed segment."""
    odtype = q.dtype
    _mark("varlen", odtype, q.shape)

    bq = cu_seqlens_q.tolist()
    bk = bq if cu_seqlens_k is None else cu_seqlens_k.tolist()

    call = dict(dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal)

    outs = []
    for i in range(len(bq) - 1):
        a, b = bq[i], bq[i + 1]
        if b - a <= 0:
            continue
        ka, kb = bk[i], bk[i + 1]
        if kb - ka <= 0:
            continue
        seg = _v100_func(_cast(q[a:b]).unsqueeze(0),
                         _cast(k[ka:kb]).unsqueeze(0),
                         _cast(v[ka:kb]).unsqueeze(0),
                         **call)
        if isinstance(seg, tuple):
            seg = seg[0]
        outs.append(seg.squeeze(0))
    if not outs:
        raise RuntimeError("flash_attn_varlen_func: no non-empty segments")
    return torch.cat(outs, dim=0).to(odtype)
