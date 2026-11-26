"""Condition-specific split conformal lower bounds for the nine braking critics.

Run ``python -m training.conformal_critics calibrate`` before online inference.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.func import functional_call, stack_module_state, vmap

from .config import Settings
from .dataset import start_groups
from .train_braking_critic import OBJECTIVE
from .visual_models import ARCHITECTURE, PREPROCESS, ResNetCritic

CONDITIONS = tuple(f'{time}_{weather}' for time in ('day', 'sunset', 'night')
                   for weather in ('sunny', 'cloudy', 'rainy'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def threshold(scores, alpha):
    """One-sided finite-sample split conformal order statistic."""
    if not 0 < alpha < 1 or not scores or not all(math.isfinite(x) for x in scores):
        raise ValueError('Expected finite calibration scores and 0 < alpha < 1')
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    if rank > len(scores):
        raise ValueError(f'{len(scores)} calibration groups cannot support alpha={alpha}; '
                         f'need at least {math.ceil(1 / alpha) - 1}')
    return sorted(scores)[rank - 1]


def calibrate(root, output, alpha):
    records = {}
    for condition in CONDITIONS:
        folder = root / f'{condition}_critic_500_s42'
        checkpoint_path = folder / 'checkpoint.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        manifest = json.loads((folder / 'split.json').read_text())
        predictions = torch.load(folder / 'test_predictions.pt', map_location='cpu', weights_only=True)
        if (checkpoint.get('objective') != OBJECTIVE or checkpoint.get('snapshot_id') != manifest['snapshot_id']
                or manifest['condition'] != condition):
            raise ValueError(f'{condition}: incompatible checkpoint or snapshot')
        data_dir = folder / 'data' / condition
        states = torch.load(data_dir / 'states.pth', weights_only=True)
        groups = start_groups(dict(states=torch.cat(states), lengths=torch.tensor([len(x) for x in states])),
                              checkpoint['settings']['start_tolerance_m'])
        train = set(manifest['split']['train'])
        test = set(manifest['split']['test'])
        if train & test or {groups[i] for i in train} & {groups[i] for i in test}:
            raise ValueError(f'{condition}: calibration and training groups overlap')
        refs, pred, target = predictions['refs'], predictions['prediction'], predictions['target']
        if refs.ndim != 2 or refs.shape[1] != 2 or len(refs) != len(pred) or pred.shape != target.shape:
            raise ValueError(f'{condition}: invalid predictions')
        if set(refs[:, 0].tolist()) != test or not torch.isfinite(pred).all() or not torch.isfinite(target).all():
            raise ValueError(f'{condition}: invalid held-out predictions')
        maxima = {}
        for episode, residual in zip(refs[:, 0].tolist(), (pred - target).tolist()):
            group = groups[episode]
            maxima[group] = max(maxima.get(group, -math.inf), residual)
        records[condition] = dict(offset=threshold(list(maxima.values()), alpha), groups=len(maxima),
                                  episodes=len(test), rows=len(refs), snapshot_id=manifest['snapshot_id'],
                                  checkpoint_sha256=digest(checkpoint_path))
    result = dict(format_version=1, alpha=alpha, score='maximum prediction minus target per start group',
                  conditions=records)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    return result


class ConformalCritics:
    """Evaluate aligned critic/action pairs; outputs remain tensors on the chosen device."""
    def __init__(self, root, calibration, device='cuda'):
        self.device = torch.device(device)
        artifact = json.loads(Path(calibration).read_text())
        if artifact.get('format_version') != 1 or set(artifact['conditions']) != set(CONDITIONS):
            raise ValueError('Invalid calibration artifact')
        models, offsets, signatures = [], [], []
        for condition in CONDITIONS:
            path = Path(root) / f'{condition}_critic_500_s42' / 'checkpoint.pt'
            record = artifact['conditions'][condition]
            if digest(path) != record['checkpoint_sha256']:
                raise ValueError(f'{condition}: checkpoint differs from calibrated checkpoint')
            ckpt = torch.load(path, map_location='cpu', weights_only=True)
            if (ckpt.get('architecture') != ARCHITECTURE or ckpt.get('preprocessing') != PREPROCESS
                    or ckpt.get('objective') != OBJECTIVE or ckpt.get('snapshot_id') != record['snapshot_id']):
                raise ValueError(f'{condition}: unsupported checkpoint')
            cfg = ckpt['settings']
            signatures.append((tuple(cfg['hidden_sizes']), cfg['visual_history_frames']))
            model = ResNetCritic(cfg['hidden_sizes'], history_frames=cfg['visual_history_frames'])
            model.load_state_dict(ckpt['critic'])
            model.eval().requires_grad_(False)
            models.append(model)
            offsets.append(record['offset'])
        if not all(math.isfinite(x) for x in offsets):
            raise ValueError('Nonfinite calibration offset')
        self.history_frames = max(signature[1] for signature in signatures)
        self.batches = []
        for signature in dict.fromkeys(signatures):
            indices = [i for i, value in enumerate(signatures) if value == signature]
            matching = [models[i] for i in indices]
            params, buffers = stack_module_state(matching)
            params = {k: v.to(self.device) for k, v in params.items()}
            buffers = {k: v.to(self.device) for k, v in buffers.items()}
            self.batches.append((indices, signature[1], matching[0].to('meta'), params, buffers))
        self.offsets = torch.tensor(offsets, device=self.device)

    @torch.inference_mode()
    def predict(self, images, actions):
        """images: [H,224,224,3] shared history or [9,H,224,224,3]; actions: [9,2]."""
        images = torch.as_tensor(images, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        shape = (self.history_frames, 224, 224, 3)
        if images.dtype != torch.uint8 or tuple(images.shape) not in (shape, (9, *shape)):
            raise ValueError(f'Expected uint8 RGB history {shape} or nine histories')
        if tuple(actions.shape) != (9, 2) or not torch.isfinite(actions).all():
            raise ValueError('Expected nine finite actions [9,2]')
        if images.ndim == 4:
            images = images.unsqueeze(0).expand(9, *shape)
        raw = torch.empty(9, device=self.device)
        for indices, history, model, params, buffers in self.batches:
            batch_images = images[indices, -history:]
            if history == 1:
                batch_images = batch_images[:, 0]
            def evaluate(params, buffers, image, action):
                return functional_call(model, (params, buffers), (image.unsqueeze(0), action.unsqueeze(0))).squeeze(0)
            with torch.backends.cudnn.flags(enabled=False):
                raw[indices] = vmap(evaluate)(params, buffers, batch_images, actions[indices])
        return raw, raw - self.offsets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['calibrate'])
    parser.add_argument('--root', type=Path, default=Path('training/_local'))
    parser.add_argument('--output', type=Path, default=Path('training/_local/conformal_braking.json'))
    parser.add_argument('--alpha', type=float, default=0.1)
    args = parser.parse_args()
    result = calibrate(args.root, args.output, args.alpha)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
