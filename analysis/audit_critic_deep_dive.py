"""Reproducible read-only audit of the cloudy 1300-episode critic.

Writes diagnostics only; does not modify data, weights, or training code.
Run: .venv/bin/python -m analysis.audit_critic_deep_dive
"""
from dataclasses import replace
import json
from pathlib import Path

import torch
from torch import nn

from training.config import Settings
from training.dataset import decision_features, first_failure, safety_margin
from training.train_braking_critic import observed_returns
from training.train_combined_braking_critics import CombinedExamples, load_sources
from training.visual_models import ResNetCritic, temporal_features


ROOT = Path('training/_local/day_cloudy_critic_1300_s42')


def summary(x):
    x = torch.as_tensor(x, dtype=torch.float32)
    return dict(n=x.numel(), mean=float(x.mean()),
                quantiles=torch.quantile(x, torch.tensor([0., .1, .5, .9, 1.])).tolist()) if x.numel() else dict(n=0)


def errors(prediction, target):
    unsafe = target <= 0
    return dict(mse=float((prediction-target).square().mean()),
                rmse=float((prediction-target).square().mean().sqrt()),
                unsafe_rows=int(unsafe.sum()),
                false_safe_rate=float((prediction[unsafe] > 0).float().mean()) if unsafe.any() else None)


def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    checkpoint = torch.load(ROOT / 'checkpoint.pt', map_location='cpu', weights_only=True)
    cfg = Settings(**checkpoint['settings'])
    audit = json.loads((ROOT / 'audit.json').read_text())
    sources = load_sources('day_cloudy', Path('dataset/combined_800'), Path('dataset/braking_lane_tracked_500_per_condition'))
    rows, source_reports = {}, {}
    for source_id, source in enumerate(sources):
        lengths, kinds, positive, count, censored, safety_sign_changes = [], {'lower': 0, 'upper': 0, 'none': 0}, 0, 0, 0, 0
        actions = []
        for local in range(source['count']):
            gaps = source['columns']['distances'][local]
            meta = None
            if source_id:
                meta = torch.load(source['folder'] / 'metadata' / f'episode_{local:03d}.pth', weights_only=True)
                gaps = torch.cat((meta['initial_gap'].reshape(1), gaps))
            first = 0 if source_id else 1
            y = observed_returns(gaps, cfg)
            undisc = observed_returns(gaps, replace(cfg, gamma=1.))
            h = safety_margin(gaps, cfg.safety_buffer_m, cfg.max_gap_m)
            hit = first_failure(h)
            kind = 'none' if hit is None else ('lower' if gaps[hit] <= cfg.safety_buffer_m else 'upper')
            kinds[kind] += 1
            if hit is not None:
                positive += int((y > 0).sum()); count += len(y)
            else:
                censored += len(y)
            safety_sign_changes += int(((y > 0) & (undisc <= 0)).sum())
            lengths.append(len(gaps)-1)
            ids = torch.arange(first, first+len(y))
            poses = source['columns']['states'][local]
            # The first braking row has no preceding saved pose: zero velocity
            # estimate there, explicitly a limitation of this diagnostic baseline.
            state = decision_features(poses[(ids-1).clamp_min(0)], poses[ids], gaps[:len(y)], cfg.dt)
            u = source['columns']['actions'][local][ids].clone(); u[:, 0].clamp_(-3, 3)
            actions.append(u)
            # No future information is included in these input features.
            x = torch.cat((state, u, torch.full((len(y), 1), float(source_id))), 1)
            rows[local + (800 if source_id else 0)] = (x, y, undisc)
        all_actions = torch.cat(actions)
        source_reports[source['name']] = dict(recorded_horizon_seconds=summary(torch.tensor(lengths)*cfg.dt),
            first_boundary_hit_episodes=kinds, rows_before_observed_hit=count,
            positive_discounted_labels_before_observed_hit=positive,
            positive_discounted_fraction_before_hit=positive/max(1, count),
            rows_from_no_hit_recordings=censored,
            signs_changed_by_discounting=safety_sign_changes,
            longitudinal=summary(all_actions[:, 0]), steering=summary(all_actions[:, 1]),
            longitudinal_saturation_fraction=float((all_actions[:, 0].abs() >= 2.999).float().mean()))
    report = dict(objective=checkpoint['objective'], settings=checkpoint['settings'], sources=source_reports,
                  history=json.loads((ROOT / 'losses.json').read_text()),
                  baseline_caveat='One seed, small MLP, same observed labels; simulator pose finite differences are privileged. Source identity is included. First braking row has zero estimated velocity.')
    splits = {}
    for name, ids in audit['split'].items():
        splits[name] = tuple(torch.cat([rows[i][j] for i in ids]) for j in (0, 1))
    report['label_distribution'] = {name: summary(y) for name, (_, y) in splits.items()}
    train_x, train_y = splits['train']; val_x, val_y = splits['validation']
    report['validation_constant_baseline'] = errors(train_y.mean().expand_as(val_y), val_y)
    mean, std = train_x.mean(0), train_x.std(0).clamp_min(.01)
    train_x, val_x = (train_x-mean)/std, (val_x-mean)/std
    model = nn.Sequential(nn.Linear(train_x.shape[1], 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    history = []
    for step in range(3001):
        if step:
            ids = torch.randint(len(train_y), (256,))
            loss = (model(train_x[ids]).squeeze(-1)-train_y[ids]).square().mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % 300 == 0:
            with torch.no_grad():
                history.append(dict(step=step, train=errors(model(train_x).squeeze(-1), train_y),
                                    validation=errors(model(val_x).squeeze(-1), val_y)))
    report['privileged_state_supervised_baseline'] = history
    print('Numerical audit complete', flush=True)
    # Check for exact repeated/black frames without assuming camera timestamps.
    image_reports = {}
    for source in sources:
        repeated, pairs, black, frames_seen, episodes = 0, 0, 0, 0, []
        repeated_gap_changes, maximum_run = [], 1
        for local in torch.linspace(0, source['count']-1, 20).long().tolist():
            frames = torch.load(source['folder'] / 'obses' / f'episode_{local:03d}.pth', mmap=True, weights_only=True)
            same = (frames[1:] == frames[:-1]).flatten(1).all(1)
            blanks = (frames == 0).flatten(1).all(1)
            gaps = source['columns']['distances'][local]
            # For original data, row i's known pre-action gap is distances[i-1].
            # Omit pair (0,1), whose first pre-action gap was not saved.
            if source['name'] == 'original':
                repeated_gap_changes.extend((gaps[1:-1]-gaps[:-2]).abs()[same[1:]].tolist())
            run = 1
            for value in same.tolist():
                run = run + 1 if value else 1
                maximum_run = max(maximum_run, run)
            repeated += int(same.sum()); pairs += len(same)
            black += int(blanks.sum()); frames_seen += len(frames)
            episodes.append(dict(episode=local, frames=len(frames), repeated_adjacent=int(same.sum()), black=int(blanks.sum())))
        image_reports[source['name']] = dict(frames=frames_seen, repeated_adjacent=repeated, adjacent_pairs=pairs,
            black=black, maximum_identical_frame_run=maximum_run,
            pre_action_gap_change_with_identical_image=summary(repeated_gap_changes), episodes=episodes)
    report['sampled_rgb_integrity'] = image_reports
    # Encode 64 held-out histories once, then test the exact learned critic head.
    dataset = CombinedExamples(sources, audit['split']['test'], cfg)
    saved = torch.load(ROOT / 'test_predictions.pt', weights_only=True)
    assert torch.equal(saved['refs'], torch.tensor(dataset.refs))
    assert torch.equal(saved['target'], dataset.targets)
    undiscounted = torch.cat([rows[i][2] for i in audit['split']['test']])
    report['held_out_predictions_against_observed_boundary_crossing'] = {
        name: errors(saved['prediction'][mask], undiscounted[mask])
        for name, mask in dict(all=torch.ones(len(undiscounted), dtype=torch.bool),
                              original=saved['refs'][:, 0] < 800,
                              braking=saved['refs'][:, 0] >= 800).items()}
    report['boundary_classification_caveat'] = 'Undiscounted observed first-hit labels indicate a boundary crossing within the recording under collected continuation, not unavoidable failure or optimal-policy safety; no-hit recordings are censored.'
    critic = ResNetCritic(cfg.hidden_sizes, freeze_batch_norm=True, history_frames=cfg.visual_history_frames)
    critic.load_state_dict(checkpoint['critic']); critic.eval()
    ids = torch.linspace(0, len(dataset)-1, 64).long()
    features, commands = [], []
    with torch.inference_mode():
        for chunk in ids.split(4):
            items = [dataset[int(i)] for i in chunk]
            images, u = torch.stack([x[0] for x in items]), torch.stack([x[1] for x in items])
            features.append(temporal_features(critic, images)); commands.append(u)
        features, commands = torch.cat(features), torch.cat(commands)
        def predict(f, u): return critic.head(torch.cat((f, u), -1)).squeeze(-1)
        q = predict(features, commands)
        def delta(other): return summary((other-q).abs())
        grid = []
        for longitudinal in (-3., 0., 3.):
            u = commands.clone(); u[:, 0] = longitudinal
            grid.append(predict(features, u))
        grid = torch.stack(grid, 1)
        temporal = features.reshape(len(ids), cfg.visual_history_frames, -1)
        report['held_out_sensitivity_probe'] = dict(n=len(ids),
            checkpoint_prediction_max_difference=float((q-saved['prediction'][ids]).abs().max()),
            longitudinal_grid_span=summary(grid.max(1).values-grid.min(1).values),
            preferred_brake_coast_throttle_counts=torch.bincount(grid.argmax(1), minlength=3).tolist(),
            zero_action_absolute_change=delta(predict(features, torch.zeros_like(commands))),
            reverse_history_absolute_change=delta(predict(temporal.flip(1).flatten(1), commands)),
            repeat_latest_frame_absolute_change=delta(predict(temporal[:, -1:].expand_as(temporal).flatten(1), commands)),
            caveat='Sensitivity measures dependence, not causal correctness or counterfactual ranking accuracy.')
    destination = Path('analysis/critic_deep_dive_20261125.json')
    destination.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(destination, flush=True)
    print(json.dumps({k:v for k,v in report.items() if k not in ('settings','history','privileged_state_supervised_baseline','sampled_rgb_integrity')}, indent=2))


if __name__ == '__main__':
    main()
