"""Experiment-only exact query chunking; never splits the softmax key axis."""
import torch
from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl, _maybe_reshape_attn_mask


def chunked_forward(self, query, key, value, attn_metadata=None, mask_mode='broadcast_k'):
    mask = None
    if attn_metadata is not None:
        mask = _maybe_reshape_attn_mask(query, key, attn_metadata.attn_mask, mask_mode=mask_mode)
    q, k, v = (x.permute(0, 2, 1, 3) for x in (query, key, value))
    if q.shape[1] != k.shape[1]:
        if q.shape[1] % k.shape[1]:
            raise ValueError('Invalid GQA heads')
        repeats = q.shape[1] // k.shape[1]
        k, v = (x.repeat_interleave(repeats, dim=1) for x in (k, v))
    output = torch.empty_like(q)
    chunk = 128
    for start in range(0, q.shape[-2], chunk):
        end = min(start + chunk, q.shape[-2])
        part_mask = mask
        if mask is not None and mask.ndim >= 2 and mask.shape[-2] != 1:
            part_mask = mask[..., start:end, :]
        if self.causal:
            # SDPA's rectangular causal convention is upper-left aligned.
            causal = torch.arange(k.shape[-2], device=q.device)[None, :] <= torch.arange(start, end, device=q.device)[:, None]
            if part_mask is None:
                part_mask = causal
            elif part_mask.dtype == torch.bool:
                part_mask = part_mask & causal
            else:
                part_mask = part_mask.masked_fill(~causal, float('-inf'))
        output[..., start:end, :] = torch.nn.functional.scaled_dot_product_attention(
            q[..., start:end, :], k, v, attn_mask=part_mask,
            dropout_p=0.0, is_causal=False, scale=self.softmax_scale,
        )
    return output.permute(0, 2, 1, 3)


def install():
    SDPAImpl._forward_impl = chunked_forward
    print('H3_QUERY_CHUNK_SDPA installed chunk=128 full-key softmax', flush=True)
