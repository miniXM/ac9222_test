"""Short-lived GPU5 VAE helper; invoked only by experiment proxy."""
import sys
import torch
from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE, MiniMaxH3AudioVAE

def cpu_tree(x):
    if isinstance(x, torch.Tensor):
        if x.is_floating_point() and not torch.isfinite(x).all():
            raise ValueError('VAE produced non-finite output')
        return x.detach().cpu()
    if isinstance(x, tuple): return tuple(cpu_tree(v) for v in x)
    if isinstance(x, list): return [cpu_tree(v) for v in x]
    return x

if __name__ == '__main__':
    kind, method, inp, out = sys.argv[1:]
    allowed = {'video': {'encode_image','encode_video','decode_latent'}, 'audio': {'encode_waveform','decode_latent'}}
    if method not in allowed[kind]: raise ValueError(method)
    # This experimental POWER9 torch/cuDNN stack reports GET engine failure
    # for video FP32 conv3d. Use PyTorch CUDA convolution fallback, not CPU.
    torch.backends.cudnn.enabled = False
    args = torch.load(inp, map_location='cpu', weights_only=False)
    cls = MiniMaxH3VideoVAE if kind == 'video' else MiniMaxH3AudioVAE
    model = cls('/srv/models/MiniMax-H3-FP8/Ref2VA/'+kind+'_vae', device=torch.device('cuda:0'))
    if method == 'decode_latent': args = (args[0].to('cuda:0'),)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16, enabled=(kind=='video' and method=='decode_latent')):
        result = getattr(model, method)(*args)
    torch.save(cpu_tree(result), out)
    print('VAE_OK', kind, method, 'peak_GiB', torch.cuda.max_memory_allocated()/2**30, flush=True)
