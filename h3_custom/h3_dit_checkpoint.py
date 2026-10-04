"""Experiment-only Diffusers-to-packed H3 loader and SM70 FP8 storage fallback.
No native FP8 GEMM: dequantize one local linear at a time, then multiply in the
per-layer GEMM dtype chosen by ``gemm_dtype`` (fp16 on the tensor cores for the
attention/MLP body, bf16/fp32 where the activations exceed fp16's 65504 range).
"""
import json
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors import safe_open
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

ROOT=Path('/srv/models/MiniMax-H3-FP8/Ref2VA/transformer')

# Control flag for the A/B dequant probe. When set, FP8-stored weights are
# dequantized to bf16 once at load instead of per-call. See SM70StoredFP8 vs
# the dequant branch in load_h3.
H3_DEQUANT_MODE = os.environ.get("H3_DEQUANT_MODE", "0") == "1"

class SM70FloatLinear(UnquantizedLinearMethod):
    def process_weights_after_loading(self, layer): pass
    def apply(self, layer, x, bias=None):
        y=F.linear(x.to(layer.weight.dtype),layer.weight,None if bias is None else bias.to(layer.weight.dtype))
        return y.to(x.dtype)

def gemm_dtype(name):
    """Per-layer GEMM dtype for the fp16 mixed-precision V100 path.

    fp16 everywhere the matmul is compute-bound and the activations are bounded
    by the power-of-two rescales; bf16/fp32 where the values are genuinely
    outside fp16's +-65504 range even after rescaling.
    """
    if name.startswith('condition_proj'):
        # Qwen3-VL hidden states project to ~96k on this very first linear.
        return torch.bfloat16
    if name.startswith('time_embedder'):
        return torch.float32
    if name.startswith('video_patch_proj') or name.startswith('audio_patch_proj'):
        return torch.float32
    if name.startswith('final_layer.video_out') or name.startswith('final_layer.audio_out'):
        return torch.float32
    if name.startswith('token_refiner.'):
        return torch.float16
    if name.startswith('blocks.') and ('.attn.' in name or '.mlp.' in name):
        return torch.float16
    # AdaLN modulations and anything else: bf16 keeps fp32's exponent range.
    return torch.bfloat16


class SM70StoredFP8(UnquantizedLinearMethod):
    """FP8-stored weight, dequantized per call into the layer's GEMM dtype.

    V100 has no FP8 cores, and keeping the dequantized copy resident would
    double the 36 GB checkpoint on the card, so the dequantization is done per
    call (a few GB/s of bandwidth, negligible next to the GEMM).
    """
    def __init__(self, dtype=torch.float16):
        self.dtype = dtype

    def process_weights_after_loading(self, layer): pass

    def apply(self, layer, x, bias=None):
        dt = self.dtype
        # Direct cast, no fp32 intermediate: measured 1.12 ms vs 1.93 ms for the
        # largest layer, and the fp32 staging tensor is 4x the fp8 bytes.
        w = layer.weight.to(dt) * layer.h3_scale.to(dt)[:, None]
        y = F.linear(x.to(dt), w, None if bias is None else bias.to(dt))
        return y.to(x.dtype)


def source_names(name):
    exact={'video_patch_proj':'proj_in','audio_patch_proj':'audio_proj_in','condition_proj':'context_embedder',
           'time_embedder.proj_in':'time_embedder.linear_1','time_embedder.proj_out':'time_embedder.linear_2',
           'final_layer.video_out':'proj_out','final_layer.audio_out':'audio_proj_out',
           'final_layer.adaln_proj.linear':'norm_out.linear', 'final_layer.norm':'norm_out.norm'}
    stem,suffix=name.rsplit('.',1)
    if stem in exact: return [exact[stem]+'.'+suffix]
    stem=stem.replace('token_refiner.blocks.','token_refiner.refiner_blocks.')
    if stem.startswith('blocks.'): stem='transformer_blocks.'+stem[len('blocks.'):]
    if stem.endswith('.attn.qkv_proj'):
        return [stem[:-len('qkv_proj')]+'to_'+q+'.'+suffix for q in ('q','k','v')]
    stem=stem.replace('.attn.out_proj','.attn.to_out.0').replace('.attn.q_norm','.attn.norm_q').replace('.attn.k_norm','.attn.norm_k')
    stem=stem.replace('.mlp.fc1','.ff.net.0.proj').replace('.mlp.fc2','.ff.net.2')
    return [stem+'.'+suffix]

class Reader:
    def __init__(self):
        self.files={}
        for f in sorted(ROOT.glob('*.safetensors')):
            with safe_open(str(f),framework='pt',device='cpu') as h:
                for k in h.keys():
                    if k in self.files: raise ValueError('duplicate '+k)
                    self.files[k]=str(f)
    def get(self,k):
        with safe_open(self.files[k],framework='pt',device='cpu') as h: return h.get_tensor(k)


def load_h3(model):
    reader=Reader()
    rank,world=get_tensor_model_parallel_rank(),get_tensor_model_parallel_world_size()
    modules=dict(model.named_modules())
    loaded=set(); used=set()
    # Validate every parameter key before allocation: no silently skipped weights.
    for name,p in model.named_parameters():
        for k in source_names(name):
            if k not in reader.files: raise ValueError('H3 mapping missing: '+name+' -> '+k)
    for name,p in model.named_parameters():
        keys=source_names(name)
        vals=[reader.get(k) for k in keys]; used.update(keys)
        parent=modules[name.rsplit('.',1)[0]]
        suffix=name.rsplit('.',1)[1]
        is_fp8=all(v.dtype==torch.float8_e4m3fn for v in vals)
        if any(v.dtype==torch.float8_e4m3fn for v in vals) and not is_fp8:
            raise ValueError('mixed packed QKV precision '+name)
        scales=None
        if is_fp8:
            scales=[reader.get(k[:-7]+'.weight_scale').float().view(-1) for k in keys]
            used.update(k[:-7]+'.weight_scale' for k in keys)
        if len(keys)==3:
            # QKV partition preserves Q_all/K_all/V_all local vLLM order.
            vals=[v.chunk(world,dim=0)[rank] for v in vals]
            value=torch.cat([v.float() for v in vals],dim=0).to(vals[0].dtype)
            if scales is not None: scale=torch.cat([s.chunk(world)[rank] for s in scales])
        else:
            value=vals[0]
            scale=scales[0] if scales is not None else None
            if '.mlp.fc1.' in name:
                gate,up=value.chunk(2,dim=0)
                value=torch.cat([gate.chunk(world,dim=0)[rank].float(),up.chunk(world,dim=0)[rank].float()],dim=0).to(value.dtype)
                if scale is not None:
                    a,b=scale.chunk(2); scale=torch.cat([a.chunk(world)[rank],b.chunk(world)[rank]])
            elif value.shape != p.shape:
                if suffix=='bias': value=value.chunk(world,dim=0)[rank]
                elif hasattr(parent,'input_is_parallel'):
                    value=value.chunk(world,dim=1)[rank]
                else:
                    value=value.chunk(world,dim=0)[rank]
                    if scale is not None: scale=scale.chunk(world)[rank]
        if tuple(value.shape)!=tuple(p.shape): raise ValueError(f'H3 shape mismatch {name}: {value.shape} != {p.shape}')
        if is_fp8:
            if H3_DEQUANT_MODE:
                # Control path (H3_DEQUANT_MODE=1): dequantize FP8 -> bf16 ONCE at
                # load, store as a plain bf16 weight and run through
                # SM70FloatLinear. This removes the per-call FP8 cast+scale
                # entirely. A revived (A)/(B) sensitivity after this switch
                # isolates the bug to the per-call dequant runtime; an unchanged
                # near-constant velocity points at the checkpoint content itself
                # (or the conditioning path), NOT the dequant math.
                dt = torch.bfloat16
                w = (value.float() * scale.float().view(-1, *([1] * (value.dim() - 1)))).to(dt)
                newp = torch.nn.Parameter(w.to(device=model.h3_device).contiguous(), requires_grad=False)
                newp.__dict__.update(p.__dict__)
                parent._parameters[suffix] = newp
                parent.quant_method = SM70FloatLinear()
                if rank == 0:
                    print('H3_DEQUANT bf16', name, 'max',
                          float(w.float().abs().max()), flush=True)
            else:
                newp=torch.nn.Parameter(value.to(device=model.h3_device).contiguous(),requires_grad=False)
                newp.__dict__.update(p.__dict__)
                parent._parameters[suffix]=newp
                parent.register_buffer('h3_scale',scale.to(model.h3_device).contiguous())
                dt=gemm_dtype(name)
                # A dequantized magnitude beyond fp16's ~65504 would become inf;
                # fall back to bf16 (same exponent range as fp32) for that layer.
                if dt==torch.float16 and float(value.float().abs().max())>60000.0:
                    dt=torch.bfloat16
                parent.quant_method=SM70StoredFP8(dt)
        else:
            # Layers the ModelOpt config left unquantized keep their declared
            # fp32 precision; bf16-declared layers follow the same mixed rule.
            dt=p.dtype if p.dtype==torch.float32 else gemm_dtype(name)
            if dt==torch.float16 and float(value.float().abs().max())>60000.0:
                dt=torch.bfloat16
            newp=torch.nn.Parameter(value.to(device=model.h3_device,dtype=dt).contiguous(),requires_grad=False)
            newp.__dict__.update(p.__dict__)
            parent._parameters[suffix]=newp
            if suffix=='weight' and hasattr(parent,'quant_method'): parent.quant_method=SM70FloatLinear()
        loaded.add(name)
        if len(loaded)%50==0: print('H3_LOAD_PROGRESS',rank,len(loaded),flush=True)
        del vals,value
    for name,b in model.named_buffers():
        if name.endswith('.h3_scale'): continue
        if name in reader.files:
            value=reader.get(name); used.add(name)
            parent=modules[name.rsplit('.',1)[0]]
            parent._buffers[name.rsplit('.',1)[1]]=value.to(model.h3_device,dtype=b.dtype)
            loaded.add(name)
        elif name=='rope.inv_freq':
            config=json.loads((ROOT/'config.json').read_text())
            theta=float(config.get('rope_theta',10000.0))
            n=model.arch.rope_inv_freq_len
            modules['rope']._buffers['inv_freq']=theta**(-torch.arange(0,2*n,2,device=model.h3_device,dtype=torch.float32)/(2*n))
            loaded.add(name)
        elif b.device.type=='meta': raise ValueError('Uninitialized H3 buffer '+name)
    extra=set(reader.files)-used
    extra={k for k in extra if not k.endswith('.input_scale')}
    if extra: raise ValueError('Unmapped checkpoint tensors: '+str(sorted(extra)[:20]))
    counts={}
    for n,m in model.named_modules():
        q=getattr(m,'quant_method',None)
        if isinstance(q,SM70StoredFP8): counts[str(q.dtype)]=counts.get(str(q.dtype),0)+1
    print('H3_LOAD_COVERAGE',len(loaded),'parameters/buffers; ignored input scales only;'
          ' fp8-storage layers by GEMM dtype',counts,flush=True)
    return loaded
