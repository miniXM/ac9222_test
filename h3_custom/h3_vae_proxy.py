"""Parameter-free VAE RPC proxy for the isolated H3 experiment."""
import os
import sys
import tempfile
import subprocess
import torch

class H3VAEProxy(torch.nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
    def is_distributed_enabled(self): return False
    def _call(self, method, *args):
        args = tuple(x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in args)
        with tempfile.TemporaryDirectory(prefix='h3_vae_') as work:
            inp, out = os.path.join(work,'input.pt'), os.path.join(work,'output.pt')
            torch.save(args,inp)
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = '5'
            subprocess.run([sys.executable,'/tmp/h3_vae_worker.py',self.kind,method,inp,out],env=env,check=True,timeout=600)
            return torch.load(out,map_location='cpu',weights_only=True)
    def encode_image(self, image): return self._call('encode_image',image)
    def encode_video(self, frames): return self._call('encode_video',frames)
    def encode_waveform(self, waveform, sample_rate): return self._call('encode_waveform',waveform,sample_rate)
    def decode_latent(self, latent): return self._call('decode_latent',latent)
