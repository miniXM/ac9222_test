"""Experimental GPU streaming encoder. No full text-model allocation."""
import os, sys, json, time
from pathlib import Path
import torch
torch.backends.cudnn.enabled = False
from safetensors import safe_open
from transformers import Qwen3VLConfig
from vllm_omni.diffusion.models.minimax_h3.encoder import (
    MiniMaxH3Qwen3VLEncoder, MiniMaxH3Qwen3VLVisionModel,
    MiniMaxH3Qwen3VLTextDecoderLayer, MiniMaxH3Qwen3VLTextRotaryEmbedding,
)
class Group:
    rank_in_group=0
    world_size=1
class Vision(MiniMaxH3Qwen3VLVisionModel):
    def forward(self, pixels, *args, **kwargs):
        return super().forward(pixels.to(torch.float16), *args, **kwargs)
class StreamText(torch.nn.Module):
    def __init__(self, config, reader, encoder):
        super().__init__()
        self.config=config
        self.reader=reader
        object.__setattr__(self, 'encoder', encoder)
        self.embed_tokens=torch.nn.Embedding(config.vocab_size,config.hidden_size,device='meta',dtype=torch.float16)
        self.embed_tokens.to_empty(device='cuda:0')
        with torch.no_grad():
            emb=reader('model.language_model.embed_tokens.weight')
            print(f'EMBED absmax={float(emb.abs().max()):.4g}',flush=True)
            self.embed_tokens.weight.copy_(emb)
        # The final RMSNorm was missing before; hidden states grew unbounded.
        self.register_buffer('final_norm_weight',reader('model.language_model.norm.weight'),persistent=False)
        print(f'FINAL_NORM absmax={float(self.final_norm_weight.abs().max()):.4g}',flush=True)
        self.rotary_emb=MiniMaxH3Qwen3VLTextRotaryEmbedding(config)
    def forward(self, hidden, positions, *, visual_pos_masks=None, deepstack_visual_embeds=None):
        print(f'HIDDEN_IN absmax={float(hidden.abs().max()):.4g}',flush=True)
        rotary=self.rotary_emb(hidden,positions)
        for i in range(50):
            t=time.monotonic()
            with torch.device('meta'):
                layer=MiniMaxH3Qwen3VLTextDecoderLayer(Group(), self.config, torch.float16)
            layer.to_empty(device='cuda:0')
            layer.to(dtype=torch.float16)
            params=dict(layer.named_parameters())
            loaded=set()
            prefix=f'model.language_model.layers.{i}.'
            for key in self.reader.keys:
                if not key.startswith(prefix) or not key.endswith('.weight'):
                    continue
                mapped=self.encoder._map_weight_name(key)
                if mapped is None: continue
                name, shard=mapped
                local=name.split(f'layers.{i}.',1)[1]
                if local not in params: raise RuntimeError(f'unmapped {key} -> {local}')
                p=params[local]
                value=self.reader(key)
                if local == 'mlp.gate_up_proj.weight':
                    size=self.config.intermediate_size
                    offset=size if shard==1 else 0
                    p.data[offset:offset+size].copy_(value)
                elif local == 'self_attn.qkv_proj.weight':
                    q=self.config.num_attention_heads*self.config.head_dim
                    kv=self.config.num_key_value_heads*self.config.head_dim
                    offset={'q':0,'k':q,'v':q+kv}[shard]
                    p.data[offset:offset+value.shape[0]].copy_(value)
                else:
                    p.data.copy_(value)
                loaded.add(local)
                del value
            missing=set(params)-loaded
            if missing: raise RuntimeError(f'missing layer {i}: {missing}')
            if i==0:
                print(f'LAYER0_NORMS in={float(params["input_layernorm.weight"].abs().max()):.4g} post={float(params["post_attention_layernorm.weight"].abs().max()):.4g} qnorm={float(params["self_attn.q_norm.weight"].abs().max()):.4g}',flush=True)
                hooks=[]
                for nm,mod in list(layer.named_modules()):
                    if not nm: continue
                    def make(nm=nm):
                        def hook(m,inp,out):
                            o=out if torch.is_tensor(out) else (out[0] if isinstance(out,(tuple,list)) and out and torch.is_tensor(out[0]) else None)
                            if o is not None:
                                print(f'  SUB {nm} absmax={float(o.abs().max()):.4g}',flush=True)
                        return hook
                    hooks.append(mod.register_forward_hook(make()))
            hidden=layer(hidden,position_embeddings=rotary)
            if i==0:
                for h in hooks: h.remove()
            if deepstack_visual_embeds is not None and i<len(deepstack_visual_embeds):
                hidden=hidden.clone()
                hidden[visual_pos_masks,:]+=deepstack_visual_embeds[i].to(hidden.device,hidden.dtype)
            if not torch.isfinite(hidden).all(): raise RuntimeError(f'nonfinite at layer {i}')
            del layer,params,p
            torch.cuda.synchronize()
            print(f'LAYER {i} seconds={time.monotonic()-t:.2f} alloc={torch.cuda.memory_allocated()/2**30:.2f}GiB hidden_absmax={float(hidden.abs().max()):.4g}',flush=True)
        # stock vllm-omni (encoder.py:752 "returning UNNORMALIZED layer-50 states")
        # runs 50 layers and returns hidden_states as-is; _map_weight_name even drops
        # "model.language_model.norm.weight" (-> None), so the final RMSNorm is NOT
        # part of the reference conditioning path. Applying it here deviates from the
        # reference and corrupts the only conditioning signal, so keep it opt-in.
        if os.environ.get('H3_TE_FINAL_NORM','0')=='1':
            eps=getattr(self.config,'rms_norm_eps',1e-6)
            rms=torch.rsqrt(hidden.float().pow(2).mean(-1,keepdim=True)+eps).to(hidden.dtype)
            hidden=hidden*rms*self.final_norm_weight
            print(f'FINAL_NORM_APPLIED absmax={float(hidden.abs().max()):.4g}',flush=True)
        else:
            print(f'LAYER50_RAW absmax={float(hidden.abs().max()):.4g} '
                  f'(unnormalized, matches stock encoder.py:752)',flush=True)
        return hidden
class Reader:
    def __init__(self,path):
        self.path=Path(path)
        self.keys=json.loads((self.path/'model.safetensors.index.json').read_text())['weight_map']
    def __call__(self,key):
        with safe_open(str(self.path/self.keys[key]),framework='pt',device='cpu') as f:
            value=f.get_tensor(key)
        if value.dtype==torch.float8_e4m3fn:
            scale_key=key[:-len('weight')]+'weight_scale'
            if scale_key not in self.keys: raise RuntimeError(f'missing scale {key}')
            with safe_open(str(self.path/self.keys[scale_key]),framework='pt',device='cpu') as f:
                scale=f.get_tensor(scale_key).float()
            if scale.numel()!=value.shape[0]: raise RuntimeError(f'invalid scale shape {key}: {scale.shape}')
            value=value.float()*scale.reshape(-1,*([1]*(value.ndim-1)))
        if not torch.isfinite(value.float()).all(): raise RuntimeError(f'nonfinite weight {key}')
        return value.to(device='cuda:0',dtype=torch.float16)
@torch.inference_mode()
def main():
    torch.cuda.set_device(0)
    path='/srv/models/MiniMax-H3-FP8/Ref2VA/text_encoder'
    config=Qwen3VLConfig.from_pretrained(path)
    reader=Reader(path)
    encoder=MiniMaxH3Qwen3VLEncoder(path,device=torch.device('cuda:0'),load_model=False,encoder_group=Group())
    with torch.device('meta'):
        encoder.vision=Vision(config.vision_config)
    encoder.vision.to_empty(device='cuda:0')
    encoder.vision.to(dtype=torch.float16)
    # This plain tensor is not moved by to_empty; rebuild outside meta scope.
    dim=encoder.vision.rotary_pos_emb.dim
    encoder.vision.rotary_pos_emb._inv_freq=1.0/(10000.0**(torch.arange(0,dim,2,dtype=torch.float32)/dim))
    vp=dict(encoder.vision.named_parameters())
    loaded=set()
    for key in reader.keys:
        if not key.startswith('model.visual.'): continue
        local=key[len('model.visual.'):]
        if local not in vp: raise RuntimeError(f'unmapped vision {key}')
        vp[local].copy_(reader(key)); loaded.add(local)
    if set(vp)-loaded: raise RuntimeError(f'missing vision {set(vp)-loaded}')
    del vp
    encoder.text_model=StreamText(config.text_config,reader,encoder)
    encoder.eval()
    if len(sys.argv)>1:
        payload=torch.load(sys.argv[1],map_location='cpu',weights_only=True)
    else:
        payload={'input_ids':torch.tensor([151643,40,123,456,789,42,13,151645])}
    hidden=encoder.encode_ids(**payload)
    print('ENCODE_OK',hidden.shape, 'finite',bool(torch.isfinite(hidden).all()),'peak_GiB',torch.cuda.max_memory_allocated()/2**30,flush=True)
    if not torch.isfinite(hidden).all(): raise RuntimeError('nonfinite encoded hidden states')
    if len(sys.argv)>2: torch.save(hidden,sys.argv[2])
if __name__=='__main__': main()
