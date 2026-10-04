import json,torch
from pathlib import Path
from safetensors import safe_open
p=Path('/srv/models/MiniMax-H3-FP8/Ref2VA/transformer')
index=json.loads((p/'diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
torch.manual_seed(42)
blocks=['token_refiner.refiner_blocks.0','transformer_blocks.5']
for block in blocks:
 weights=[]; scales=[]
 for q in ('q','k','v'):
  key=block+'.attn.to_'+q+'.weight'
  with safe_open(str(p/index[key]),framework='pt',device='cpu') as f:
   view=f.get_slice(key); shape=view.get_shape(); n=shape[0]//4
   pieces=[view[r*n:r*n+32,:256] for r in range(4)]
  sk=key[:-len('weight')]+'weight_scale'
  with safe_open(str(p/index[sk]),framework='pt',device='cpu') as f: s=f.get_tensor(sk).float().reshape(-1)
  weights.append(pieces); scales.append([s[r*n:r*n+32] for r in range(4)])
 for rank in range(4):
  raw=torch.cat([weights[j][rank].float() for j in range(3)]).cuda()
  scale=torch.cat([scales[j][rank] for j in range(3)]).cuda()
  x=torch.randn(32,256,device='cuda',dtype=torch.float16)
  w=raw.half()*scale.half()[:,None]
  got=torch.nn.functional.linear(x,w)
  ref=torch.nn.functional.linear(x.float(),raw*scale[:,None])
  rel=float((got.float()-ref).norm()/ref.norm())
  assert torch.isfinite(got).all()
  assert rel<0.01,(block,rank,rel)
  separate=torch.cat([torch.nn.functional.linear(x,(weights[j][rank].cuda().half()*scales[j][rank].cuda().half()[:,None])) for j in range(3)],dim=-1)
  assert torch.equal(got,separate)
  print('QKV_GPU_PASS',block,'rank',rank,'relative_l2',rel,'packed_vs_separate_equal',True,flush=True)
print('PROBE_COMPLETE limitations=sampled_rows_256_columns_not_full_model',flush=True)
