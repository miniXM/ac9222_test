"""Run the streaming text encoder twice (fp16 and bf16) on GPU5 and save the
encoded hidden states plus statistics, so the DiT-side conditioning scale can
be audited on CPU afterwards.

Short-lived single-GPU process; does not touch the production service.
"""
import os
import sys
import json
import time
from pathlib import Path

import torch

torch.backends.cudnn.enabled = False
from safetensors import safe_open
from transformers import Qwen3VLConfig
from vllm_omni.diffusion.models.minimax_h3.encoder import (
    MiniMaxH3Qwen3VLEncoder,
    MiniMaxH3Qwen3VLVisionModel,
    MiniMaxH3Qwen3VLTextDecoderLayer,
    MiniMaxH3Qwen3VLTextRotaryEmbedding,
)

PATH = "/srv/models/MiniMax-H3-FP8/Ref2VA/text_encoder"
OUTDIR = "/tmp/h3_te_dtype_test"
os.makedirs(OUTDIR, exist_ok=True)


class Group:
    rank_in_group = 0
    world_size = 1


class Vision(MiniMaxH3Qwen3VLVisionModel):
    def forward(self, pixels, *args, **kwargs):
        return super().forward(pixels.to(DTYPE), *args, **kwargs)


DTYPE = torch.float16
_report = {}


@torch.inference_mode()
def run_once(dtype, tag):
    global DTYPE
    DTYPE = dtype
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    config = Qwen3VLConfig.from_pretrained(PATH)
    reader = Reader(dtype)
    encoder = MiniMaxH3Qwen3VLEncoder(
        PATH, device=torch.device("cuda:0"), load_model=False, encoder_group=Group()
    )
    with torch.device("meta"):
        encoder.vision = Vision(config.vision_config)
    encoder.vision.to_empty(device="cuda:0")
    encoder.vision.to(dtype=dtype)
    dim = encoder.vision.rotary_pos_emb.dim
    encoder.vision.rotary_pos_emb._inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    )
    vp = dict(encoder.vision.named_parameters())
    loaded = set()
    for key in reader.keys:
        if not key.startswith("model.visual."):
            continue
        local = key[len("model.visual."):]
        vp[local].copy_(reader(key))
        loaded.add(local)
    if set(vp) - loaded:
        raise RuntimeError(f"missing vision {set(vp) - loaded}")
    del vp
    encoder.text_model = StreamText(config.text_config, reader, encoder, dtype)
    encoder.eval()
    payload = {"input_ids": torch.tensor([151643, 40, 123, 456, 789, 42, 13, 151645])}
    t0 = time.monotonic()
    hidden = encoder.encode_ids(**payload).float().cpu()
    rec = {
        "tag": tag,
        "dtype": str(dtype),
        "shape": list(hidden.shape),
        "std": round(float(hidden.std()), 5),
        "absmax": round(float(hidden.abs().max()), 4),
        "finite": bool(torch.isfinite(hidden).all()),
        "seconds": round(time.monotonic() - t0, 1),
    }
    print("TE_RESULT", json.dumps(rec), flush=True)
    _report[tag] = rec
    torch.save(hidden, os.path.join(OUTDIR, f"hidden_{tag}.pt"))
    del encoder, config, reader
    torch.cuda.empty_cache()
    return hidden


class Reader:
    def __init__(self, dtype):
        self.path = Path(PATH)
        self.keys = json.loads((self.path / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        self.dtype = dtype

    def __call__(self, key):
        with safe_open(str(self.path / self.keys[key]), framework="pt", device="cpu") as f:
            value = f.get_tensor(key)
        if value.dtype == torch.float8_e4m3fn:
            scale_key = key[: -len("weight")] + "weight_scale"
            with safe_open(str(self.path / self.keys[scale_key]), framework="pt", device="cpu") as f:
                scale = f.get_tensor(scale_key).float()
            value = value.float() * scale.reshape(-1, *([1] * (value.ndim - 1)))
        return value.to(device="cuda:0", dtype=self.dtype)


class StreamText(torch.nn.Module):
    def __init__(self, config, reader, encoder, dtype):
        super().__init__()
        self.config = config
        self.reader = reader
        object.__setattr__(self, "encoder", encoder)
        self.embed_tokens = torch.nn.Embedding(
            config.vocab_size, config.hidden_size, device="meta", dtype=dtype
        )
        self.embed_tokens.to_empty(device="cuda:0")
        with torch.no_grad():
            emb = reader("model.language_model.embed_tokens.weight")
            self.embed_tokens.weight.copy_(emb)
        self.register_buffer(
            "final_norm_weight", reader("model.language_model.norm.weight"), persistent=False
        )
        self.rotary_emb = MiniMaxH3Qwen3VLTextRotaryEmbedding(config)
        self.dtype = dtype
        self.layer_stats = []

    def forward(self, hidden, positions, *, visual_pos_masks=None, deepstack_visual_embeds=None):
        rotary = self.rotary_emb(hidden, positions)
        for i in range(50):
            with torch.device("meta"):
                layer = MiniMaxH3Qwen3VLTextDecoderLayer(Group(), self.config, self.dtype)
            layer.to_empty(device="cuda:0")
            layer.to(dtype=self.dtype)
            params = dict(layer.named_parameters())
            loaded = set()
            prefix = f"model.language_model.layers.{i}."
            for key in self.reader.keys:
                if not key.startswith(prefix) or not key.endswith(".weight"):
                    continue
                mapped = self.encoder._map_weight_name(key)
                if mapped is None:
                    continue
                name, shard = mapped
                local = name.split(f"layers.{i}.", 1)[1]
                p = params[local]
                value = self.reader(key)
                if local == "mlp.gate_up_proj.weight":
                    size = self.config.intermediate_size
                    offset = size if shard == 1 else 0
                    p.data[offset : offset + size].copy_(value)
                elif local == "self_attn.qkv_proj.weight":
                    q = self.config.num_attention_heads * self.config.head_dim
                    kv = self.config.num_key_value_heads * self.config.head_dim
                    offset = {"q": 0, "k": q, "v": q + kv}[shard]
                    p.data[offset : offset + value.shape[0]].copy_(value)
                else:
                    p.data.copy_(value)
                loaded.add(local)
                del value
            missing = set(params) - loaded
            if missing:
                raise RuntimeError(f"missing layer {i}: {missing}")
            hidden = layer(hidden, position_embeddings=rotary)
            if deepstack_visual_embeds is not None and i < len(deepstack_visual_embeds):
                hidden = hidden.clone()
                hidden[visual_pos_masks, :] += deepstack_visual_embeds[i].to(
                    hidden.device, hidden.dtype
                )
            if not torch.isfinite(hidden).all():
                raise RuntimeError(f"nonfinite at layer {i}")
            self.layer_stats.append(
                {"layer": i, "absmax": round(float(hidden.abs().max()), 4),
                 "std": round(float(hidden.float().std()), 4)}
            )
            del layer, params
        eps = getattr(self.config, "rms_norm_eps", 1e-6)
        rms = torch.rsqrt(hidden.float().pow(2).mean(-1, keepdim=True) + eps).to(hidden.dtype)
        hidden = hidden * rms * self.final_norm_weight
        return hidden


if __name__ == "__main__":
    run_once(torch.float16, "fp16")
    run_once(torch.bfloat16, "bf16")
    with open(os.path.join(OUTDIR, "report.json"), "w") as f:
        json.dump(_report, f, indent=2)
    print("TE_DTYPE_TEST_DONE", flush=True)
