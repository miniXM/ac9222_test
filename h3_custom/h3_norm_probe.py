import json,torch
from pathlib import Path
from safetensors import safe_open
p=Path('/srv/models/MiniMax-H3-FP8/Ref2VA/transformer')
idx=json.loads((p/'diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
torch.manual_seed(42)
for b in ('token_refiner.refiner_blocks.0','transformer_blocks.5'):
 key=b+'.attn.norm_q.weight'
 if key not in idx:
  print('AVAILABLE_NORM_KEYS',[k for k in idx if b in k and 'norm' in k],flush=True)
  continue
 with safe_open(str(p/idx[key]),framework='pt',device='cpu') as f: w=f.get_tensor(key).float().cuda()
 for amp in (0.001,0.01,0.1,1.0,10.0):
  x=torch.randn(32,w.numel(),device='cuda')*amp
  ref=torch.nn.functional.rms_norm(x,(w.numel(),),w,1e-5)
  old=torch.nn.functional.rms_norm(x/16,(w.numel(),),w,1e-5)
  fixed=torch.nn.functional.rms_norm(x/16,(w.numel(),),w,1e-5/256)
  rel=lambda z:float((z-ref).norm()/ref.norm())
  print('NORM_TEST',b,'input_rms',amp,'old_rel',rel(old),'fixed_rel',rel(fixed),flush=True)
  assert rel(fixed)<1e-5
print('PROBE_DONE synthetic_inputs_actual_norm_weights_not_full_attention',flush=True)
