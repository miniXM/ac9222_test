"""Experiment-only NaN localisation. Reports the first non-finite tensor."""
import torch


def install():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3DiTModel, MiniMaxH3DiTBlock, MiniMaxH3FinalLayer,
        MiniMaxH3TokenRefiner, MiniMaxH3Attention, MiniMaxH3TimeEmbedder,
    )

    def stats(name, t):
        if not torch.is_tensor(t):
            return f'{name}=<{type(t).__name__}>'
        if torch.isfinite(t).all():
            return f'{name} ok {tuple(t.shape)} {t.dtype} absmax={float(t.abs().max()):.4g}'
        return (f'{name} BAD {tuple(t.shape)} {t.dtype} '
                f'nan={int(torch.isnan(t).sum())} inf={int(torch.isinf(t).sum())}')

    def scan(tag, obj):
        seq = obj if isinstance(obj, (tuple, list)) else (obj,)
        bad = [stats(f'{tag}.o{j}', t) for j, t in enumerate(seq)
               if torch.is_tensor(t) and not torch.isfinite(t).all()]
        return bad

    counter = {'block': 0}

    for cls, tag in ((MiniMaxH3Attention, 'attn'), (MiniMaxH3TokenRefiner, 'refiner'),
                     (MiniMaxH3FinalLayer, 'final'), (MiniMaxH3TimeEmbedder, 'time')):
        original = cls.forward

        def make(original=original, tag=tag):
            def wrapper(self, *a, **kw):
                out = original(self, *a, **kw)
                bad = scan(tag, out)
                if bad:
                    print('H3_NAN ' + ' | '.join(bad), flush=True)
                    raise RuntimeError('H3_NAN at ' + tag)
                return out
            return wrapper
        cls.forward = make()

    original_block = MiniMaxH3DiTBlock.forward

    def block_forward(self, *a, **kw):
        index = counter['block']
        counter['block'] += 1
        out = original_block(self, *a, **kw)
        bad = scan(f'block{index}', out)
        if bad:
            print('H3_NAN ' + ' | '.join(bad), flush=True)
            raise RuntimeError(f'H3_NAN at block {index}')
        if index == 0:
            seq = out if isinstance(out, (tuple, list)) else (out,)
            print('H3_DIAG block0 ' + ' | '.join(
                stats(f'o{j}', t) for j, t in enumerate(seq) if torch.is_tensor(t)), flush=True)
        return out
    MiniMaxH3DiTBlock.forward = block_forward

    original_model = MiniMaxH3DiTModel.forward

    def model_forward(self, **kw):
        counter['block'] = 0
        print('H3_DIAG in ' + ' | '.join(
            stats(k, v) for k, v in kw.items() if torch.is_tensor(v)), flush=True)
        out = original_model(self, **kw)
        bad = scan('model', out)
        if bad:
            print('H3_NAN ' + ' | '.join(bad), flush=True)
            raise RuntimeError('H3_NAN at model output')
        print('H3_DIAG out ' + ' | '.join(
            stats(f'o{j}', t) for j, t in enumerate(out if isinstance(out, (tuple, list)) else (out,))
            if torch.is_tensor(t)), flush=True)
        return out
    MiniMaxH3DiTModel.forward = model_forward
    print('H3_NAN_PROBE installed', flush=True)
