"""Matched PID and raw-critic full-brake intervention tests on held-out routes."""

import argparse
from collections import deque
from contextlib import suppress
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from run_gap_pid import GapPID
from .braking_scenario import BrakingScenario, InvalidBrakingScenario
from .collect_braking import sample_state, start_server
from .config import Settings
from .evaluate_closed_loop import write_json
from .train_braking_critic import OBJECTIVE
from .visual_models import ARCHITECTURE, PREPROCESS, ResNetCritic


def filtered_action(nominal, score, enabled):
    if not math.isfinite(score):
        raise ValueError('Critic produced a nonfinite decision score')
    intervention = enabled and score <= 0.
    return (-3., float(nominal[1])) if intervention else tuple(nominal), intervention


def check_initial_state(actual, expected):
    for key in ('pose', 'velocity'):
        if not torch.allclose(torch.tensor(actual[key]), torch.tensor(expected[key]), rtol=0., atol=1e-3):
            raise RuntimeError(f'Paired initial {key} differs by more than 0.001')


def run_rollout(env, entry, model, cfg, device, gains, duration, enabled, reference=None, conformal=None, trajectory=None, selection=None):
    pid = GapPID(entry['target_gap'], *gains)
    first = sample_state(env, cfg.dt)
    initial = dict(pose=first[2].tolist(), velocity=first[4].tolist(), gap_m=first[3])
    if reference is not None:
        check_initial_state(initial, reference)
    if conformal is not None:
        conformal.reset()
    history_frames = conformal.predictor.history_frames if conformal else cfg.visual_history_frames
    history = deque(maxlen=history_frames)
    image = env.get_rgb_observation(first[0]).clone()
    history.extend([image] * history_frames)
    horizon = math.ceil(entry['brake_after'] / cfg.dt) + math.ceil(duration / cfg.dt)
    trace, ending, failure = [], 'braking_complete', None
    sample = first
    for tick in range(horizon):
        tick_started = time.monotonic()
        try:
            env.validate_lanes()
        except InvalidBrakingScenario as error:
            ending, failure = 'lane_failure', str(error)
            break
        if conformal is None:
            nominal = env.nominal_action(pid)
            observation = torch.stack(list(history)) if cfg.visual_history_frames > 1 else history[-1]
            with torch.inference_mode():
                score = float(model(observation.unsqueeze(0).to(device),
                                    torch.tensor([nominal], dtype=torch.float32, device=device))[0])
            action, intervened = filtered_action(nominal, score, enabled)
            env.render_decision(score, 'safety_filter' if enabled else 'pid', intervened)
            details = {}
        else:
            from .conformal_display import show_bounds
            from .conformal_critics import CONDITIONS
            steer = -env.tracker(env.ego).steering()
            actions, gaps, values, bounds = conformal.decide(torch.stack(list(history)), steer, cfg.dt)
            if selection not in (None, "raw_min", "lower_min", "raw_max", "lower_max"):
                raise ValueError("Unsupported selection rule")
            ranking = bounds if selection in ("lower_min", "lower_max") else values
            if not all(math.isfinite(v) for v in values + bounds):
                raise ValueError("Nonfinite critic output")
            chosen = (max if selection.endswith("max") else min)(range(len(ranking)), key=ranking.__getitem__) if selection else 0
            nominal = action = actions[chosen]
            score, intervened = values[chosen], False
            details = dict(selected=CONDITIONS[chosen], selection=selection, specialists={name: dict(action=actions[i], gap_estimate_m=gaps[i],
                           q_m=values[i], lower_bound_m=bounds[i]) for i, name in enumerate(CONDITIONS)})
            if env.display is not None:
                show_bounds(env.display, env.current_chase, values, bounds, trajectory)
        env.step(action)
        successor = sample_state(env, cfg.dt)
        if successor[0] != sample[0] + 1:
            raise RuntimeError('Nonconsecutive evaluation sensor frames')
        trace.append(dict(frame=sample[0], elapsed_s=tick * cfg.dt, gap_m=sample[3],
                          ego_speed_mps=float(sample[4][:3].norm()),
                          leader_speed_mps=float(sample[4][3:].norm()),
                          nominal_action=list(nominal), action=list(action), q_m=score,
                          intervened=intervened, next_gap_m=successor[3], **details))
        time.sleep(.02)
        sample = successor
        history.append(env.get_rgb_observation(sample[0]).clone())
        trace[-1]["work_s"] = time.monotonic() - tick_started
        if getattr(env.args, 'external_pacing', False):
            time.sleep(max(0., cfg.dt - (time.monotonic() - tick_started)))
        trace[-1]['tick_wall_s'] = time.monotonic() - tick_started
        if any(first[0] <= frame <= sample[0] for frame, _ in env.events):
            ending = 'collision'
            break
        if any(other != env.ego.id for _, other in env.leader_events):
            ending, failure = 'leader_contact', 'Leader contacted an object outside the following pair'
            break
    measurements = env.measurements()
    collisions = [(frame, other) for frame, other in measurements['collision_events']
                  if first[0] <= frame <= sample[0]]
    if collisions:
        first_hit = min(frame for frame, _ in collisions)
        trace = [row for row in trace if row['frame'] < first_hit]
        ending = 'collision'
    # Include a terminal lane check at the fixed-horizon endpoint as well.
    if ending == 'braking_complete':
        try:
            env.validate_lanes()
        except InvalidBrakingScenario as error:
            ending, failure = 'lane_failure', str(error)
    alerts = [row for row in trace if row['intervened']]
    if env.display is not None:
        status = {'collision': 'Collision', 'lane_failure': 'Lane departure',
                  'leader_contact': 'Leader contact', 'braking_complete': 'Braking complete'}[ending]
        env.display.show(env.current_chase, None, 'safety_filter' if enabled else 'pid', False,
                         status_override=status)
        time.sleep(1.)
    return dict(mode='filtered' if enabled else 'pid', initial=initial, ending=ending,
                collision=bool(collisions), lane_failure=ending == 'lane_failure', failure=failure,
                first_intervention_s=alerts[0]['elapsed_s'] if alerts else None,
                intervention_steps=len(alerts), transitions=len(trace),
                minimum_gap_m=min([first[3]] + [row['next_gap_m'] for row in trace]),
                collision_events=collisions, trace=trace)


def summarize(pairs, safety_buffer_m=1., max_gap_m=9., modes=('pid', 'filtered')):
    valid = [pair for pair in pairs if pair['status'] == 'complete']
    result = dict(requested_pairs=len(pairs), completed_pairs=len(valid),
                  failed_pairs=[dict(episode=pair['episode'], error=pair['error'])
                                for pair in pairs if pair['status'] != 'complete'])
    for mode in modes:
        rows = [pair[mode] for pair in valid]
        result[mode] = dict(episodes=len(rows), collisions=sum(row['collision'] for row in rows),
                            lane_failures=sum(row['lane_failure'] for row in rows),
                            leader_contacts=sum(row['ending'] == 'leader_contact' for row in rows),
                            completed_braking=sum(row['ending'] == 'braking_complete' for row in rows),
                            episodes_with_intervention=sum(row['intervention_steps'] > 0 for row in rows),
                            lower_buffer_violations=sum(row['minimum_gap_m'] <= safety_buffer_m for row in rows),
                            upper_gap_violations=sum(row['initial']['gap_m'] >= max_gap_m or
                                                     any(step['next_gap_m'] >= max_gap_m for step in row['trace'])
                                                     for row in rows))
    if set(modes) != {'pid', 'filtered'}:
        return result
    result['paired_collision_outcomes'] = dict(
        pid_collision_filtered_no_collision=sum(pair['pid']['collision'] and not pair['filtered']['collision'] for pair in valid),
        filtered_collision_pid_no_collision=sum(pair['filtered']['collision'] and not pair['pid']['collision'] for pair in valid),
        both_collision=sum(pair['pid']['collision'] and pair['filtered']['collision'] for pair in valid),
        neither_collision=sum(not pair['pid']['collision'] and not pair['filtered']['collision'] for pair in valid))
    result['paired_completion_outcomes'] = dict(
        pid_failure_filtered_complete=sum(pair['pid']['ending'] != 'braking_complete' and
                                          pair['filtered']['ending'] == 'braking_complete' for pair in valid),
        pid_complete_filtered_failure=sum(pair['pid']['ending'] == 'braking_complete' and
                                          pair['filtered']['ending'] != 'braking_complete' for pair in valid))
    advances = []
    for pair in valid:
        alarm = pair['filtered']['first_intervention_s']
        full_brake = [step['elapsed_s'] for step in pair['pid']['trace'] if step['action'][0] <= -3. + 1e-6]
        if alarm is not None and full_brake:
            advances.append(full_brake[0] - alarm)
    result['intervention_timing'] = dict(
        episodes_compared=len(advances),
        median_advance_over_pid_full_brake_s=float(torch.tensor(advances).median()) if advances else None,
        episodes_intervening_at_or_after_pid_full_brake=sum(value <= 0 for value in advances))
    return result


def save_plot(summary, output, modes=('pid', 'filtered')):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(figsize=(7, 4.5))
    base = [0] * len(modes)
    labels = {'pid': 'PID', 'filtered': 'Critic with full-brake fallback'}
    for field, label, color in (('collisions', 'Collision', '#b84539'),
                                ('lane_failures', 'Lane departure', '#d59e35'),
                                ('completed_braking', 'Completed braking', '#398570')):
        values = [summary[mode][field] for mode in modes]
        axes.bar([labels[mode] for mode in modes], values, bottom=base, label=label, color=color, width=.6)
        for index, value in enumerate(values):
            if value:
                axes.text(index, base[index] + value / 2, str(value), ha='center', va='center', color='white')
        base = [previous + value for previous, value in zip(base, values)]
    axes.set(ylabel='Held-out scenarios', ylim=(0, max(base) + 2))
    axes.legend(loc='upper center', bbox_to_anchor=(.5, 1.18), ncol=3, frameon=False)
    figure.tight_layout()
    figure.savefig(output / 'outcomes.png', dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conformal', type=Path, help='Show nine specialist values and calibrated lower bounds')
    parser.add_argument('--critic-root', type=Path, default=Path('training/_local'))
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--port', type=int, default=2800)
    parser.add_argument('--start-server', action='store_true')
    parser.add_argument('--server', type=Path, default=Path('.local/carla-0.9.15/CarlaUE4.sh'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--n-scenarios', type=int, help='Run this many held-out scenarios, in saved split order')
    parser.add_argument('--modes', nargs='+', choices=('pid', 'filtered'), default=['pid', 'filtered'])
    parser.add_argument('--graphics', choices=('vulkan', 'windowed'), default='vulkan')
    parser.add_argument('--real-time', action='store_true', help='Pace each complete decision/physics tick to simulation time')
    args = parser.parse_args()
    if args.n_scenarios is not None and args.n_scenarios < 1:
        parser.error('--n-scenarios must be positive')
    if args.conformal and args.modes != ['pid']:
        parser.error('--conformal requires --modes pid (no safety filter)')
    if len(set(args.modes)) != len(args.modes):
        parser.error('--modes must be unique')
    if args.outdir.exists() and not args.resume:
        parser.error('Output exists. Use a fresh directory or --resume')
    if args.resume and not (args.outdir / 'manifest.json').exists():
        parser.error('--resume requires an evaluation manifest')
    if args.start_server and not args.server.is_file():
        parser.error('CARLA server launcher is missing')
    checkpoint_path = args.pilot / 'checkpoint.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    split = json.loads((args.pilot / 'split.json').read_text())
    test_ids = split['split']['test']
    if args.n_scenarios is not None:
        if args.n_scenarios > len(test_ids):
            parser.error('--n-scenarios exceeds the held-out split size')
        test_ids = test_ids[:args.n_scenarios]
    if (checkpoint['objective'] != OBJECTIVE or checkpoint['architecture'] != ARCHITECTURE or
            checkpoint['preprocessing'] != PREPROCESS or checkpoint['snapshot_id'] != split['snapshot_id']):
        parser.error('Pilot critic does not match the snapshot and supported input contract')
    source = Path(split['source'])
    collector = json.loads((source.parent / 'manifest.json').read_text())['settings']
    cfg = Settings(**checkpoint['settings'])
    gains = (collector['kp'], collector['ki'], collector['kd'])
    args.outdir.mkdir(parents=True, exist_ok=args.resume)
    manifest = dict(pilot=str(args.pilot.resolve()), snapshot_id=split['snapshot_id'], test_ids=test_ids,
                    checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                    nominal=('Day-sunny camera specialist PID; nine proposals evaluated' if args.conformal
                             else 'True bumper-gap PID with persisted route tracking'), pid_gains=gains,
                    threshold_m=0., fallback='Full brake, preserve nominal route-following steering',
                    release_rule='Recompute each tick, return to PID when nominal-action Q is positive',
                    paired_initial_tolerance=1e-3, brake_duration_s=collector['brake_duration'], port=args.port,
                    modes=args.modes, graphics=args.graphics, real_time=args.real_time,
                    **(dict(conformal=str(args.conformal.resolve())) if args.conformal else {}))
    if args.resume:
        previous = json.loads((args.outdir / 'manifest.json').read_text())
        previous.setdefault('modes', ['pid', 'filtered'])
        previous.setdefault('graphics', 'vulkan')
        previous.setdefault('real_time', False)
        if json.loads(json.dumps(manifest)) != previous:
            parser.error('Resume evaluation settings differ')
    else:
        write_json(args.outdir / 'manifest.json', manifest)
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNetCritic(cfg.hidden_sizes, history_frames=cfg.visual_history_frames).to(device).eval().requires_grad_(False)
    model.load_state_dict(checkpoint['critic'])
    if any(not value.isfinite().all() for value in model.state_dict().values()):
        raise ValueError('Critic weights are nonfinite')
    conformal = None
    if args.conformal:
        from .conformal_display import ConformalDisplay
        conformal = ConformalDisplay(args.critic_root, args.conformal, device)
    import carla
    from collect_parallel import stop
    process, world, env = None, None, None
    pairs = []
    try:
        if args.start_server:
            process = start_server(args)
        client = carla.Client('127.0.0.1', args.port)
        client.set_timeout(120.)
        world = client.load_world('Town03')
        for episode in test_ids:
            destination = args.outdir / f'episode_{episode:03d}.json'
            if args.resume and destination.exists():
                previous = json.loads(destination.read_text())
                if previous['status'] == 'complete':
                    pairs.append(previous)
                    continue
            metadata = torch.load(args.pilot / 'data' / split['condition'] / 'metadata' / f'episode_{episode:03d}.pth',
                                  map_location='cpu', weights_only=True)
            entry = metadata['plan']
            rollout = SimpleNamespace(condition=entry['condition'], blueprint='vehicle.audi.a2',
                                      initial_gap=entry['initial_gap'], initial_speed=entry['initial_speed'],
                                      leader_brake_test=True, leader_brake_after=entry['brake_after'],
                                      leader_brake_duration=collector['brake_duration'], stationary_leader_test=False,
                                      graphics=args.graphics, camera=True, real_time=args.real_time,
                                      external_pacing=args.real_time or args.graphics == 'windowed',
                                      handoff_gap_bounds=collector['gap_range'], handoff_speed_bounds=collector['speed_range'])
            pair = dict(episode=episode, source_ending=metadata['summary']['ending'], status='failed')
            print(f'Evaluating held-out episode {episode}, scenario {len(pairs) + 1}/{len(test_ids)}', flush=True)
            for attempt in range(1, 4):
                try:
                    for mode in args.modes:
                        env = BrakingScenario(client, rollout, cfg.dt, entry['scenario'], reload_world=False)
                        try:
                            if env.display is not None:
                                env.display.override_label = 'Full brake'
                                env.display.pygame.display.set_caption(
                                    f'HJ critic | trajectory {len(pairs) + 1}/{len(test_ids)} | {mode}')
                            reference = pair[args.modes[0]]['initial'] if mode != args.modes[0] else None
                            pair[mode] = run_rollout(env, entry, model, cfg, device, gains,
                                                    collector['brake_duration'], mode == 'filtered', reference, conformal,
                                                    f'{len(pairs) + 1}/{len(test_ids)}')
                        finally:
                            env.close()
                            env = None
                    pair['status'] = 'complete'
                    pair.pop('error', None)
                    break
                except (RuntimeError, ValueError) as error:
                    pair['error'] = str(error)
                    print(f'Pair attempt {attempt} failed: {error}', flush=True)
            write_json(destination, pair)
            pairs.append(pair)
            write_json(args.outdir / 'summary.json', summarize(pairs, cfg.safety_buffer_m, cfg.max_gap_m, args.modes))
            if pair['status'] == 'complete':
                print(', '.join(f"{mode} {pair[mode]['ending']}" for mode in args.modes), flush=True)
        summary = summarize(pairs, cfg.safety_buffer_m, cfg.max_gap_m, args.modes)
        write_json(args.outdir / 'complete.json', summary)
        save_plot(summary, args.outdir, args.modes)
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        if env is not None:
            env.close()
        if world is not None:
            with suppress(RuntimeError):
                settings = world.get_settings()
                settings.synchronous_mode = False
                world.apply_settings(settings)
        if process is not None:
            stop(process)


if __name__ == '__main__':
    main()
