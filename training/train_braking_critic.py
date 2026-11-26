"""Actor-free RGB critic pilot using observed first-hit returns from braking data."""

import argparse
from collections import OrderedDict, defaultdict
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from generate_dataset import save_atomic
from .config import Settings, setup_device
from .dataset import first_failure, safety_margin, start_groups
from .evaluate_closed_loop import write_json
from .prepare_data import pack_episodes, read_conditions
from .train_ddpg import hj_target
from .visual_models import ARCHITECTURE, PREPROCESS, ResNetCritic


OBJECTIVE = 'recorded_policy_discounted_first_hit_observed_horizon'


def observed_returns(gaps, cfg):
    """Apply the existing HJ backup backwards through the observed continuation."""
    margins = safety_margin(gaps, cfg.safety_buffer_m, cfg.max_gap_m)
    hit = first_failure(margins)
    end = len(gaps) - 1 if hit is None else hit
    values = margins[:end + 1].clone()
    for row in reversed(range(end)):
        values[row] = hj_target(margins[row], margins[row + 1], margins[row + 1] <= 0,
                                values[row + 1], cfg.gamma)
    return values[:-1]


def grouped_split(states, train_count, seed):
    data = dict(states=torch.cat(states), lengths=torch.tensor([len(x) for x in states]))
    groups = defaultdict(list)
    for episode, group in enumerate(start_groups(data, Settings().start_tolerance_m)):
        groups[group].append(episode)
    groups = list(groups.values())
    random.Random(seed).shuffle(groups)
    ends = [0]
    for group in groups:
        ends.append(ends[-1] + len(group))
    cut = min(range(1, len(groups)), key=lambda index: abs(ends[index] - train_count))
    return dict(train=sorted(i for group in groups[:cut] for i in group),
                test=sorted(i for group in groups[cut:] for i in group))


def freeze_data(source, output, count, train_count, seed):
    with TemporaryDirectory(prefix='snapshot_', dir=output) as temporary:
        staging = Path(temporary)
        _, columns, metadata, manifest = write_snapshot(source, staging, count, train_count, seed)
        (staging / 'data').rename(output / 'data')
        (staging / 'split.json').rename(output / 'split.json')
    return output / 'data' / source.name, columns, metadata, manifest


def write_snapshot(source, output, count, train_count, seed):
    """Snapshot a committed prefix while the collector appends new episodes."""
    lengths = torch.load(source / 'seq_lengths.pth', map_location='cpu', weights_only=True)
    if len(lengths) < count:
        raise ValueError(f'Requested {count} episodes, only {len(lengths)} committed')
    lengths = lengths[:count].clone()
    folder = output / 'data' / source.name
    (folder / 'obses').mkdir(parents=True)
    (folder / 'metadata').mkdir()
    columns = {}
    digest = hashlib.sha256()
    for field in ('states', 'actions', 'distances', 'costs'):
        values = torch.load(source / f'{field}.pth', map_location='cpu', weights_only=True)
        if len(values) < count:
            raise ValueError(f'Missing committed {field} rows')
        columns[field] = [value.clone() for value in values[:count]]
        if any(len(value) != length for value, length in zip(columns[field], lengths.tolist())):
            raise ValueError(f'Committed lengths disagree with {field}')
        for value in columns[field]:
            digest.update(value.contiguous().numpy().tobytes())
        save_atomic(columns[field], folder / f'{field}.pth')
    save_atomic(lengths, folder / 'seq_lengths.pth')
    metadata = []
    for episode in range(count):
        name = f'episode_{episode:03d}.pth'
        item = torch.load(source / 'metadata' / name, map_location='cpu', weights_only=True)
        if len(item['frames']) != int(lengths[episode]) + 1:
            raise ValueError(f'Incorrect observation metadata length for episode {episode}')
        digest.update(item['frames'].numpy().tobytes())
        digest.update(item['initial_gap'].numpy().tobytes())
        metadata.append(item)
        # Committed collector episode files are immutable. Hard links preserve
        # this prefix without duplicating the RGB storage during collection.
        for kind in ('obses', 'metadata'):
            os.link(source / kind / name, folder / kind / name)
    pack_episodes(read_conditions(output / 'data'))
    split = grouped_split(columns['states'], train_count, seed)
    manifest = dict(source=str(source.resolve()), episodes=count, split=split, seed=seed,
                    snapshot_id=digest.hexdigest(), condition=source.name,
                    grouping='Both starting vehicle positions within 0.25 m stay in the same split',
                    storage='Copied numerical prefix, hard-linked immutable RGB and episode metadata')
    write_json(output / 'split.json', manifest)
    return folder, columns, metadata, manifest


class BrakingExamples(Dataset):
    def __init__(self, folder, columns, metadata, episodes, cfg):
        self.folder, self.columns, self.metadata, self.cfg = folder, columns, metadata, cfg
        self.cache, self.refs, self.targets = OrderedDict(), [], []
        for episode in episodes:
            gaps = torch.cat((metadata[episode]['initial_gap'].reshape(1), columns['distances'][episode]))
            target = observed_returns(gaps, cfg)
            self.refs.extend((episode, row) for row in range(len(target)))
            self.targets.extend(target.tolist())
        self.targets = torch.tensor(self.targets, dtype=torch.float32)
        if not self.refs:
            raise ValueError('Split has no pre-failure decision rows')

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, index):
        episode, row = self.refs[index]
        if episode not in self.cache:
            frames = torch.load(self.folder / 'obses' / f'episode_{episode:03d}.pth',
                                mmap=True, map_location='cpu', weights_only=True)
            if frames.dtype != torch.uint8 or frames.shape != (len(self.columns['actions'][episode]), 224, 224, 3):
                raise ValueError(f'Invalid RGB episode {episode}')
            self.cache[episode] = frames
            if len(self.cache) > 8:
                self.cache.popitem(last=False)
        self.cache.move_to_end(episode)
        indices = torch.arange(row - self.cfg.visual_history_frames + 1, row + 1).clamp_min(0)
        image = self.cache[episode][indices]
        if self.cfg.visual_history_frames == 1:
            image = image[0]
        return image, self.columns['actions'][episode][row], self.targets[index]


def audit(columns, metadata, ids):
    rows = [metadata[i]['summary'] for i in ids]
    return dict(episodes=len(ids), transitions=sum(len(columns['actions'][i]) for i in ids),
                collisions=sum(row['ending'] == 'collision' for row in rows),
                completed_braking=sum(row['ending'] == 'braking_complete' for row in rows),
                collisions_before_braking=sum(row['collision_before_braking'] for row in rows))


def scores(target, prediction):
    unsafe = target <= 0
    false_safe = unsafe & (prediction > 0)
    false_alarm = ~unsafe & (prediction <= 0)
    return dict(rows=len(target), mae_m=float((target - prediction).abs().mean()),
                rmse_m=float((target - prediction).square().mean().sqrt()),
                unsafe_rows=int(unsafe.sum()), safe_rows=int((~unsafe).sum()),
                false_safe_rate=float(false_safe.sum() / unsafe.sum()) if unsafe.any() else None,
                false_alarm_rate=float(false_alarm.sum() / (~unsafe).sum()) if (~unsafe).any() else None)


@torch.inference_mode()
def predict(model, data, batch_size, device):
    model.eval()
    predictions = []
    for image, action, _ in DataLoader(data, batch_size=batch_size):
        predictions.append(model(image.to(device), action.to(device)).cpu())
    return torch.cat(predictions)


def episode_scores(data, predictions, cfg):
    grouped = defaultdict(list)
    for (episode, row), prediction in zip(data.refs, predictions.tolist()):
        grouped[episode].append((row, prediction))
    records = []
    for episode, values in grouped.items():
        metadata = data.metadata[episode]
        gaps = torch.cat((metadata['initial_gap'].reshape(1), data.columns['distances'][episode]))
        buffer_failure = bool((safety_margin(gaps, cfg.safety_buffer_m, cfg.max_gap_m) <= 0).any())
        frames = metadata['frames'].tolist()
        collision_frames = [frame for frame, _ in metadata['collision_events']]
        collision_time = (min(collision_frames) - frames[0]) * cfg.dt if collision_frames else None
        alarms = [row for row, value in values if value <= 0]
        alarm_time = alarms[0] * cfg.dt if alarms else None
        records.append(dict(episode=episode, **{key: metadata['summary'][key] for key in
                            ('ending', 'collision_before_braking', 'initial_gap_m', 'requested_speed_mps')},
                            observed_buffer_failure=buffer_failure, minimum_gap_m=float(gaps.min()),
                            first_nonpositive_time_s=alarm_time, collision_time_s=collision_time,
                            warning_lead_s=collision_time - alarm_time
                            if collision_time is not None and alarm_time is not None else None))
    leads = [row['warning_lead_s'] for row in records if row['warning_lead_s'] is not None]
    return dict(episodes=records, collision_episodes=sum(row['collision_time_s'] is not None for row in records),
                collisions_detected=sum(row['warning_lead_s'] is not None for row in records),
                completed_braking_episodes=sum(row['ending'] == 'braking_complete' for row in records),
                completed_braking_with_alert=sum(row['ending'] == 'braking_complete' and
                                                 row['first_nonpositive_time_s'] is not None for row in records),
                buffer_safe_completed_episodes=sum(row['ending'] == 'braking_complete' and
                                                   not row['observed_buffer_failure'] for row in records),
                buffer_safe_completed_with_alert=sum(row['ending'] == 'braking_complete' and
                                                      not row['observed_buffer_failure'] and
                                                      row['first_nonpositive_time_s'] is not None for row in records),
                median_warning_lead_s=float(torch.tensor(leads).median()) if leads else None)


def save_plot(output, test, prediction, losses):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([row['epoch'] for row in losses], [row['train_mse'] for row in losses])
    axes[0].set(xlabel='Epoch', ylabel='Training MSE (m²)')
    axes[1].scatter(test.targets, prediction, s=5, alpha=.3)
    axes[1].axhline(0, color='black', linewidth=.8)
    axes[1].axvline(0, color='black', linewidth=.8)
    axes[1].set(xlabel='Observed first-hit return (m)', ylabel='Predicted return (m)')
    figure.tight_layout()
    figure.savefig(output / 'diagnostics.png', dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('dataset/braking_lane_tracked_500_per_condition/day_sunny'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=256)
    parser.add_argument('--train-episodes', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--history-frames', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if not 0 < args.train_episodes < args.episodes or min(args.epochs, args.batch_size, args.history_frames) < 1:
        parser.error('Require positive training settings and a nonempty held-out split')
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error('Learning rate must be positive and finite')
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = replace(Settings(), seed=args.seed, visual_history_frames=args.history_frames,
                  critic_lr=args.lr, batch_size=args.batch_size, shared_conditions=(args.source.name,),
                  train_fraction=args.train_episodes / args.episodes, val_fraction=0., dataset_selection='new')
    split_path = args.output / 'split.json'
    if split_path.exists():
        manifest = json.loads(split_path.read_text())
        if (manifest['source'], manifest['episodes'], manifest['seed']) != (str(args.source.resolve()), args.episodes, args.seed):
            parser.error('Output snapshot settings differ from this request')
        if len(manifest['split']['train']) != args.train_episodes:
            parser.error('Existing snapshot uses a different training episode count')
        folder = args.output / 'data' / manifest['condition']
        columns = {field: torch.load(folder / f'{field}.pth', weights_only=True)
                   for field in ('states', 'actions', 'distances', 'costs')}
        metadata = [torch.load(folder / 'metadata' / f'episode_{i:03d}.pth', weights_only=True)
                    for i in range(args.episodes)]
    else:
        folder, columns, metadata, manifest = freeze_data(args.source, args.output, args.episodes,
                                                         args.train_episodes, args.seed)
    train = BrakingExamples(folder, columns, metadata, manifest['split']['train'], cfg)
    test = BrakingExamples(folder, columns, metadata, manifest['split']['test'], cfg)
    report = dict(objective=OBJECTIVE, settings=asdict(cfg), snapshot_id=manifest['snapshot_id'],
                  training_epochs=args.epochs, split=manifest['split'],
                  train=audit(columns, metadata, manifest['split']['train']),
                  test=audit(columns, metadata, manifest['split']['test']),
                  target_contract='Observed recorded-policy continuation, absorbing first buffer hit, terminal observed margin at recording end',
                  input_contract='Past/current RGB frames and recorded requested action',
                  selection='Fixed epoch count chosen before held-out evaluation',
                  limitations=['Action coverage follows the true-gap PID behavior policy',
                               'Observed-horizon labels end at the recording cutoff',
                               'Closed-loop intervention performance requires separate simulator evaluation'])
    report['train']['critic_rows'], report['test']['critic_rows'] = len(train), len(test)
    write_json(args.output / 'audit.json', report)
    print(json.dumps({key: report[key] for key in ('train', 'test')}, indent=2), flush=True)
    if args.prepare_only:
        return
    if (args.output / 'checkpoint.pt').exists():
        parser.error('Checkpoint already exists. Choose a fresh output directory for another training run')
    torch.set_num_threads(4)
    device = setup_device(args.seed)
    model = ResNetCritic(cfg.hidden_sizes, pretrained=True, freeze_batch_norm=True,
                         history_frames=cfg.visual_history_frames).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == 'cuda')
    losses = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        squared_error, started = 0., time.monotonic()
        for batch_number, (image, action, target) in enumerate(loader, 1):
            image, action, target = image.to(device), action.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=device.type == 'cuda'):
                prediction = model(image, action)
                loss = F.mse_loss(prediction, target)
            if not torch.isfinite(loss):
                raise RuntimeError('Critic training loss became nonfinite')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            squared_error += float(loss.detach()) * len(target)
            if batch_number % 100 == 0:
                print(f'Epoch {epoch}, batch {batch_number}/{len(loader)}, '
                      f'elapsed {time.monotonic() - started:.1f} s', flush=True)
        row = dict(epoch=epoch, train_mse=squared_error / len(train), elapsed_s=time.monotonic() - started)
        losses.append(row)
        write_json(args.output / 'losses.json', losses)
        print(json.dumps(row), flush=True)
    prediction = predict(model, test, args.batch_size, device)
    if not torch.isfinite(prediction).all():
        raise RuntimeError('Held-out critic predictions became nonfinite')
    constant = torch.full_like(test.targets, float(train.targets.mean()))
    report['metrics'] = dict(critic=scores(test.targets, prediction), training_mean=scores(test.targets, constant))
    warnings = episode_scores(test, prediction, cfg)
    mean_warnings = episode_scores(test, constant, cfg)
    report['warning_summary'] = {name: {key: value for key, value in item.items() if key != 'episodes'}
                                 for name, item in (('critic', warnings), ('training_mean', mean_warnings))}
    write_json(args.output / 'test_episodes.json', warnings['episodes'])
    write_json(args.output / 'report.json', report)
    save_atomic(dict(format_version=1, architecture=ARCHITECTURE, preprocessing=PREPROCESS,
                     objective=OBJECTIVE, settings=asdict(cfg), snapshot_id=manifest['snapshot_id'],
                     critic=model.cpu().state_dict(), optimizer=optimizer.state_dict(),
                     epochs=args.epochs), args.output / 'checkpoint.pt')
    save_atomic(dict(refs=torch.tensor(test.refs), prediction=prediction, target=test.targets),
                args.output / 'test_predictions.pt')
    save_plot(args.output, test, prediction, losses)
    print(json.dumps(report['metrics'], indent=2), flush=True)


if __name__ == '__main__':
    main()
