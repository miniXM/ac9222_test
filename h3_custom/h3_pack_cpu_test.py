import importlib.util,torch
p='/srv/omni/src/vllm-omni-0.26.0/vllm_omni/diffusion/models/minimax_h3/packed_tokens.py'
s=importlib.util.spec_from_file_location('h3_packing_only',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
for ps in ((1,2,2),(2,2,2),(1,1,1)):
 x=torch.arange(2*24*4*8*10,dtype=torch.float32).reshape(2,24,4,8,10)
 rows=m.minimax_h3_patchify_video_latent(x,patch_size=ps)
 y=m.minimax_h3_unpatchify_video_tokens(rows,latent_shape=(4//ps[0],8//ps[1],10//ps[2],24),patch_size=ps)
 assert torch.equal(x,y)
 print('VIDEO_PACK_ROUNDTRIP_PASS',ps,tuple(rows.shape),flush=True)
a=torch.arange(2*32*20,dtype=torch.float32).reshape(2,32,20)
r=m.minimax_h3_pack_audio_latent(a)
b=m.minimax_h3_unpack_audio_tokens(r,audio_t=40,audio_channel=2)
assert torch.equal(a,b)
print('AUDIO_PACK_ROUNDTRIP_PASS no_CUDA no_model',flush=True)
