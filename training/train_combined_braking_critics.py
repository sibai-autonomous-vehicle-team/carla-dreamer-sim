"""Retrain the nine observed-return RGB critics on the 800+500 datasets.

Training is staged next to each current directory. Existing outputs are renamed
only after a complete checkpoint and held-out predictions have been written.
"""
import argparse
from collections import OrderedDict, defaultdict
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from .config import Settings
from .dataset import start_groups
from .train_braking_critic import OBJECTIVE, observed_returns, scores
from .visual_models import ARCHITECTURE, PREPROCESS, ResNetCritic

CONDITIONS = tuple(f'{t}_{w}' for t in ('day', 'sunset', 'night') for w in ('sunny', 'cloudy', 'rainy'))
FIELDS = ('states', 'actions', 'distances', 'costs')


def write_json(path, obj):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def load_sources(condition, old_root, new_root):
    sources = []
    for name, root, count in (('original', old_root, 800), ('braking', new_root, 500)):
        folder = root / condition
        lengths = torch.load(folder / 'seq_lengths.pth', map_location='cpu', weights_only=True)
        if len(lengths) != count:
            raise ValueError(f'{folder}: expected {count} episodes, got {len(lengths)}')
        columns = {field: torch.load(folder / f'{field}.pth', map_location='cpu', weights_only=True) for field in FIELDS}
        if any(len(items) != count for items in columns.values()):
            raise ValueError(f'{folder}: episode count mismatch')
        for i, length in enumerate(lengths.tolist()):
            if length < 1 or any(len(columns[field][i]) != length or not torch.isfinite(columns[field][i]).all() for field in FIELDS):
                raise ValueError(f'{folder}: invalid episode {i}')
        sources.append(dict(name=name, folder=folder.resolve(), count=count, lengths=lengths, columns=columns))
    return sources


def split_groups(sources, seed, fraction):
    """Split each source approximately 80/20, respecting shared start groups.

    Original groups are assigned first. Braking groups sharing an original
    group inherit its partition, and remaining braking groups are assigned to
    approach the braking source's target without splitting a group.
    """
    states = sum((source['columns']['states'] for source in sources), [])
    groups = start_groups(dict(states=torch.cat(states), lengths=torch.tensor([len(x) for x in states])),
                          Settings().start_tolerance_m)
    source_groups = [defaultdict(list), defaultdict(list)]
    for episode, group in enumerate(groups):
        source_groups[0 if episode < 800 else 1][group].append(episode)
    rng = random.Random(seed)
    assignments = {}
    original = list(source_groups[0].items())
    rng.shuffle(original)
    prefix = [0]
    for _, ids in original:
        prefix.append(prefix[-1] + len(ids))
    target = round(800 * fraction)
    cut = min(range(1, len(original)), key=lambda i: abs(prefix[i] - target))
    for i, (group, _) in enumerate(original):
        assignments[group] = 'train' if i < cut else 'test'
    braking = list(source_groups[1].items())
    rng.shuffle(braking)
    fixed_train = sum(len(ids) for group, ids in braking if assignments.get(group) == 'train')
    unassigned = [(group, ids) for group, ids in braking if group not in assignments]
    prefix = [0]
    for _, ids in unassigned:
        prefix.append(prefix[-1] + len(ids))
    cut = min(range(len(unassigned) + 1), key=lambda i: abs(fixed_train + prefix[i] - round(500 * fraction)))
    for i, (group, _) in enumerate(unassigned):
        assignments[group] = 'train' if i < cut else 'test'
    split = {name: [i for i, group in enumerate(groups) if assignments[group] == name] for name in ('train', 'test')}
    if not all(split.values()):
        raise ValueError('Could not create nonempty group-disjoint splits')
    return split, groups


def add_validation_split(split, groups, seed, fraction=.125):
    """Reserve ~1/8 of the training 80%, preserving the existing test IDs."""
    grouped = defaultdict(list)
    for episode in split['train']:
        grouped[groups[episode]].append(episode)
    candidates = list(grouped.values())
    random.Random(seed + 1).shuffle(candidates)
    totals = [sum(i < 800 for i in split['train']), sum(i >= 800 for i in split['train'])]
    counts = [0, 0]
    choices = []
    for cut, group in enumerate(candidates, 1):
        counts[0] += sum(i < 800 for i in group)
        counts[1] += sum(i >= 800 for i in group)
        if cut < len(candidates):
            error = sum(((counts[j] - totals[j] * fraction) / max(1, totals[j])) ** 2 for j in (0, 1))
            choices.append((error, cut))
    if not choices:
        raise ValueError('Insufficient training groups for validation')
    cut = min(choices)[1]
    validation = sorted(i for group in candidates[:cut] for i in group)
    training = sorted(i for group in candidates[cut:] for i in group)
    return dict(train=training, validation=validation, test=split['test'])


class CombinedExamples(Dataset):
    def __init__(self, sources, ids, cfg):
        self.sources, self.cfg, self.refs, targets = sources, cfg, [], []
        self.cache = OrderedDict()
        for index in ids:
            source_id = 0 if index < 800 else 1
            source = sources[source_id]
            local = index if source_id == 0 else index - 800
            gaps = source['columns']['distances'][local]
            if source_id == 1:
                meta = torch.load(source['folder'] / 'metadata' / f'episode_{local:03d}.pth', map_location='cpu', weights_only=True)
                gaps = torch.cat((meta['initial_gap'].reshape(1), gaps))
                first_row = 0
            else:
                # The archive has no initial gap. RGB/action row 1 is the first
                # row with an available pre-action gap, distances[0].
                first_row = 1
            values = observed_returns(gaps, cfg)
            self.refs.extend((index, first_row + row) for row in range(len(values)))
            targets.extend(values.tolist())
        if not self.refs: raise ValueError('No eligible decision rows')
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self): return len(self.refs)

    def __getitem__(self, item):
        index, row = self.refs[item]
        source = self.sources[0 if index < 800 else 1]
        local = index if index < 800 else index - 800
        if index not in self.cache:
            frames = torch.load(source['folder'] / 'obses' / f'episode_{local:03d}.pth', mmap=True, map_location='cpu', weights_only=True)
            expected = (int(source['lengths'][local]), 224, 224, 3)
            if frames.dtype != torch.uint8 or tuple(frames.shape) != expected: raise ValueError(f'Invalid RGB {source["folder"]}/{local}')
            self.cache[index] = frames
            if len(self.cache) > 8: self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        indices = torch.arange(row - self.cfg.visual_history_frames + 1, row + 1).clamp_min(0)
        frames = self.cache[index][indices]
        if self.cfg.visual_history_frames == 1: frames = frames[0]
        # The original collector can log requests outside actuator saturation.
        action = source['columns']['actions'][local][row].clone()
        action[0].clamp_(-3., 3.)
        return frames, action, self.targets[item]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=CONDITIONS)
    p.add_argument('--old-root', type=Path, default=Path('dataset/combined_800'))
    p.add_argument('--new-root', type=Path, default=Path('dataset/braking_lane_tracked_500_per_condition'))
    p.add_argument('--root', type=Path, default=Path('training/_local'))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--log-every', type=int, default=100, help='Print training progress every N batches')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--train-fraction', type=float, default=.8)
    p.add_argument('--history-frames', type=int, default=4)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--cpu', action='store_true', help='Allow slow CPU training')
    args = p.parse_args()
    if len(set(args.conditions)) != len(args.conditions) or min(args.epochs, args.batch_size, args.history_frames, args.log_every) < 1 or not math.isfinite(args.lr) or args.lr <= 0 or not 0 < args.train_fraction < 1: p.error('Invalid training arguments')
    if not args.prepare_only and not torch.cuda.is_available() and not args.cpu: p.error('CUDA unavailable; use --cpu only if intentional')
    torch.set_num_threads(4)
    for condition in args.conditions:
        destination = args.root / f'{condition}_critic_1300_s42'
        stage = args.root / f'{condition}_critic_1300_s42_staging'
        initialization_source = destination if destination.exists() else args.root / f'{condition}_critic_500_s42'
        previous = torch.load(initialization_source / 'checkpoint.pt', map_location='cpu', weights_only=True)
        if previous.get('objective') != OBJECTIVE or previous.get('architecture') != ARCHITECTURE or previous.get('preprocessing') != PREPROCESS:
            raise ValueError(f'{condition}: unsupported original critic')
        cfg = replace(Settings(**previous['settings']), visual_history_frames=args.history_frames)
        sources = load_sources(condition, args.old_root, args.new_root)
        split, groups = split_groups(sources, args.seed, args.train_fraction)
        split = add_validation_split(split, groups, args.seed)
        train = CombinedExamples(sources, split['train'], cfg)
        validation = CombinedExamples(sources, split['validation'], cfg)
        test = CombinedExamples(sources, split['test'], cfg)
        # Fingerprint numerical data, metadata used for targets, and split.
        h = hashlib.sha256()
        for source in sources:
            for field in FIELDS:
                for tensor in source['columns'][field]: h.update(tensor.contiguous().numpy().tobytes())
        h.update(json.dumps(split, sort_keys=True).encode())
        for i in range(500):
            meta = torch.load(sources[1]['folder'] / 'metadata' / f'episode_{i:03d}.pth', weights_only=True)
            h.update(meta['initial_gap'].contiguous().numpy().tobytes())
        snapshot_id = h.hexdigest()
        audit = dict(condition=condition, sources=[dict(name=s['name'], path=str(s['folder']), episodes=s['count']) for s in sources],
                     split=split, group_ids=groups, seed=args.seed, snapshot_id=snapshot_id,
                     train_rows=len(train), validation_rows=len(validation), test_rows=len(test), epochs=args.epochs, history_frames=cfg.visual_history_frames,
                     split_counts={name: {'original': sum(i < 800 for i in ids), 'braking': sum(i >= 800 for i in ids)} for name, ids in split.items()},
                     alignment='Original: RGB/action row i with distances[i-1] and successor distances[i], i>=1; braking: recorded initial gap then post-action gaps',
                     warning='Source datasets retain different behavior policies and observed horizons; labels are not counterfactual returns.')
        used = {episode for dataset in (train, validation, test) for episode, _ in dataset.refs}
        audit['episodes_without_eligible_rows'] = [i for i in range(1300) if i not in used]
        print(json.dumps({k: audit[k] for k in ('condition', 'split_counts', 'history_frames', 'train_rows', 'test_rows', 'snapshot_id')}), flush=True)
        if args.prepare_only: continue
        if stage.exists():
            preserved = stage.with_name(f'{stage.name}_interrupted_{time.time_ns()}')
            stage.rename(preserved)
            print(f'Preserved previous staging directory at {preserved}; restarting training from initialization', flush=True)
        stage.mkdir(parents=True)
        write_json(stage / 'audit.json', audit)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
        model = ResNetCritic(cfg.hidden_sizes, pretrained=True, freeze_batch_norm=True, history_frames=cfg.visual_history_frames).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
        scaler = torch.amp.GradScaler(device.type, enabled=device.type == 'cuda')
        history = []
        best_validation = math.inf
        best_epoch = None
        for epoch in range(1, args.epochs + 1):
            model.train(); total = 0.; started = time.monotonic()
            print(f'{condition} | epoch {epoch}/{args.epochs} | {len(loader)} batches', flush=True)
            for batch_index, (images, actions, targets) in enumerate(loader, 1):
                images, actions, targets = images.to(device), actions.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device.type, enabled=device.type == 'cuda'):
                    loss = F.mse_loss(model(images, actions), targets)
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                total += float(loss.detach()) * len(targets)
                if batch_index % args.log_every == 0 or batch_index == len(loader):
                    elapsed = time.monotonic() - started
                    remaining = elapsed / batch_index * (len(loader) - batch_index)
                    print(f'{condition} | epoch {epoch}/{args.epochs} | '
                          f'batch {batch_index}/{len(loader)} ({100 * batch_index / len(loader):.1f}%) | '
                          f'batch MSE {float(loss.detach()):.5f} | '
                          f'elapsed {elapsed:.1f}s | epoch ETA {remaining:.1f}s', flush=True)
            model.eval()
            validation_squared_error = 0.
            with torch.inference_mode():
                for images, actions, targets in DataLoader(validation, batch_size=args.batch_size):
                    prediction = model(images.to(device), actions.to(device))
                    validation_squared_error += float(F.mse_loss(prediction, targets.to(device), reduction='sum'))
            validation_mse = validation_squared_error / len(validation)
            if not math.isfinite(validation_mse):
                raise RuntimeError('Nonfinite validation loss')
            if validation_mse < best_validation:
                best_validation, best_epoch = validation_mse, epoch
                torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, stage / 'best_weights.pt')
            history.append(dict(epoch=epoch, train_mse=total / len(train), validation_mse=validation_mse,
                                best_validation_mse=best_validation, best_epoch=best_epoch, elapsed_s=time.monotonic()-started))
            write_json(stage / 'losses.json', history)
            print(condition, history[-1], flush=True)
        model.load_state_dict(torch.load(stage / 'best_weights.pt', map_location=device, weights_only=True))
        model.eval(); preds = []
        with torch.inference_mode():
            for images, actions, _ in DataLoader(test, batch_size=args.batch_size):
                preds.append(model(images.to(device), actions.to(device)).cpu())
        predictions = torch.cat(preds)
        if not torch.isfinite(predictions).all(): raise RuntimeError('Nonfinite held-out predictions')
        torch.save(dict(refs=torch.tensor(test.refs), prediction=predictions, target=test.targets), stage / 'test_predictions.pt')
        write_json(stage / 'report.json', dict(held_out=scores(test.targets, predictions), snapshot_id=snapshot_id, best_epoch=best_epoch, best_validation_mse=best_validation))
        torch.save(dict(format_version=1, architecture=ARCHITECTURE, preprocessing=PREPROCESS, objective=OBJECTIVE,
                        settings=asdict(cfg), snapshot_id=snapshot_id, critic=model.cpu().state_dict(), epochs=args.epochs,
                        best_epoch=best_epoch, best_validation_mse=best_validation, training_sources=audit['sources']), stage / 'checkpoint.pt')
        # Preserve the complete prior directory (including snapshots/evaluations).
        backup = args.root / f'{condition}_critic_1300_s42_previous_{time.time_ns()}'
        if backup.exists(): raise FileExistsError(backup)
        had_destination = destination.exists()
        if had_destination:
            os.rename(destination, backup)
        try: os.rename(stage, destination)
        except BaseException:
            if had_destination:
                os.rename(backup, destination)
            raise
        print(f'Installed {destination}; previous directory preserved at {backup if had_destination else initialization_source}', flush=True)


if __name__ == '__main__': main()
