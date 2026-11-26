"""Fresh, reproducible paired maximum-score specialist selection experiments."""
import argparse
from collections import Counter
from contextlib import suppress
import json
from pathlib import Path
import random
from types import SimpleNamespace

import torch

from .braking_scenario import BrakingScenario, InvalidBrakingScenario
from .collect_braking import start_server
from .config import Settings
from .conformal_critics import CONDITIONS, digest
from .conformal_display import ConformalDisplay
from .evaluate_braking_critic import run_rollout
from .evaluate_closed_loop import write_json, make_scenarios


def summarize(records):
    valid = [x for x in records if x['status'] == 'complete']
    result = {'attempted_pairs': len(records), 'completed_pairs': len(valid),
              'failed_pairs': [{'condition': x['condition'], 'episode': x['episode'], 'error': x.get('error')} for x in records if x['status'] != 'complete'], 'by_condition': {}}
    for condition in CONDITIONS:
        pairs = [x for x in valid if x['condition'] == condition]
        result['by_condition'][condition] = {'pairs': len(pairs)}
        for mode in ('raw_max', 'lower_max'):
            rows = [x[mode] for x in pairs]
            steps = [step for row in rows for step in row['trace']]
            result['by_condition'][condition][mode] = {'collisions': sum(r['collision'] for r in rows),
                'lane_failures': sum(r['lane_failure'] for r in rows),
                'braking_complete': sum(r['ending'] == 'braking_complete' for r in rows),
                'selections': dict(Counter(s['selected'] for s in steps)),
                'ticks': len(steps), 'ticks_over_100ms': sum(s['work_s'] > .1 for s in steps)}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outdir', type=Path, required=True)
    p.add_argument('--root', type=Path, default=Path('training/_local'))
    p.add_argument('--calibration', type=Path, default=Path('training/_local/conformal_braking.json'))
    p.add_argument('--seed', type=int, default=20261124)
    p.add_argument('--count', type=int, default=300)
    p.add_argument('--port', type=int, default=2800)
    p.add_argument('--server', type=Path, default=Path('.local/carla-0.9.15/CarlaUE4.sh'))
    p.add_argument('--start-server', action='store_true')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.count < 1: p.error('count must be positive')
    if args.outdir.exists() and not args.resume: p.error('output exists; use --resume')
    if args.resume and not (args.outdir / 'manifest.json').exists(): p.error('missing manifest for resume')
    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest = dict(seed=args.seed, count_per_condition=args.count,
                    sampling='Fresh seeded CARLA lane-route plans; uniform gap, speed and brake delay in collector ranges',
                    modes=['raw_max', 'lower_max'], selection='maximum predicted return or calibrated lower bound',
                    calibration_sha256=digest(args.calibration), critic_sha256={c: digest(args.root / f'{c}_critic_500_s42/checkpoint.pt') for c in CONDITIONS},
                    specialist_sha256={c: {f: digest(Path('specialists') / c / f) for f in ('cnn.pth','pid.json')} for c in CONDITIONS},
                    warning='Provisional calibration uses recorded PID data and may not cover selected policies or cross-condition inputs.', port=args.port)
    manifest_path = args.outdir / 'manifest.json'
    if args.resume:
        if json.loads(manifest_path.read_text()) != manifest: p.error('manifest differs on resume')
    else: write_json(manifest_path, manifest)
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictor = ConformalDisplay(args.root, args.calibration, device)
    import carla
    from collect_parallel import stop
    process = world = env = None
    records = []
    try:
        if args.start_server: process = start_server(args)
        client = carla.Client('127.0.0.1', args.port)
        client.set_timeout(120.)
        world = client.load_world('Town03')
        plans_path = args.outdir / 'scenario_plans.json'
        if plans_path.exists():
            plans = json.loads(plans_path.read_text())
        else:
            plans = {}
            source_manifest = json.loads(Path('dataset/braking_lane_tracked_500_per_condition/manifest.json').read_text())['settings']
            for index, condition in enumerate(CONDITIONS):
                route_seed = args.seed + 1009 * index
                rng = random.Random(route_seed)
                routes = make_scenarios(world, args.count, route_seed, 80)
                plans[condition] = [dict(episode=i, condition=condition, scenario=route,
                    initial_gap=rng.uniform(*source_manifest['gap_range']),
                    initial_speed=rng.uniform(*source_manifest['speed_range']),
                    target_gap=source_manifest['target_gap'],
                    brake_after=rng.uniform(*source_manifest['brake_delay_range']))
                    for i, route in enumerate(routes)]
            write_json(plans_path, plans)
        for condition in CONDITIONS:
            split = json.loads((args.root / f'{condition}_critic_500_s42' / 'split.json').read_text())
            folder = args.root / f'{condition}_critic_500_s42'
            checkpoint = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=True)
            cfg = Settings(**checkpoint['settings'])
            collector = json.loads((Path(split['source']).parent / 'manifest.json').read_text())['settings']
            gains = (collector['kp'], collector['ki'], collector['kd'])
            (args.outdir / condition).mkdir(exist_ok=True)
            for entry in plans[condition]:
                episode = entry['episode']
                path = args.outdir / condition / f'episode_{episode:03d}.json'
                if args.resume and path.exists():
                    old = json.loads(path.read_text())
                    if old['status'] == 'complete':
                        records.append(old)
                        continue
                    entry = old['plan']
                record = dict(condition=condition, episode=episode, plan=entry, status='failed', rejected_plans=[])
                print(f'{len(records)+1}/{len(CONDITIONS)*args.count} {condition} episode {episode}', flush=True)
                for attempt in range(1, 21):
                    rollout = SimpleNamespace(condition=condition, blueprint='vehicle.audi.a2',
                        initial_gap=entry['initial_gap'], initial_speed=entry['initial_speed'],
                        leader_brake_test=True, leader_brake_after=entry['brake_after'],
                        leader_brake_duration=collector['brake_duration'], stationary_leader_test=False,
                        graphics='vulkan', camera=True, real_time=False, external_pacing=False,
                        handoff_gap_bounds=collector['gap_range'], handoff_speed_bounds=collector['speed_range'])
                    try:
                        for mode in ('raw_max', 'lower_max'):
                            env = BrakingScenario(client, rollout, cfg.dt, entry['scenario'], reload_world=False)
                            try:
                                reference = record['raw_max']['initial'] if mode == 'lower_max' else None
                                record[mode] = run_rollout(env, entry, None, cfg, device, gains, collector['brake_duration'], False,
                                                           reference=reference, conformal=predictor, selection=mode)
                            finally:
                                env.close(); env = None
                        record['status'] = 'complete'; record.pop('error', None)
                        break
                    except InvalidBrakingScenario as exc:
                        record['rejected_plans'].append(dict(plan=entry, error=str(exc)))
                        seed = args.seed + 10000019 + CONDITIONS.index(condition)*1000003 + episode*101 + attempt
                        rng = random.Random(seed)
                        entry = dict(episode=episode, condition=condition,
                            scenario=make_scenarios(world, 1, seed, 80)[0],
                            initial_gap=rng.uniform(*collector['gap_range']),
                            initial_speed=rng.uniform(*collector['speed_range']),
                            target_gap=collector['target_gap'],
                            brake_after=rng.uniform(*collector['brake_delay_range']), replacement_seed=seed)
                        record.update(plan=entry, error=str(exc))
                        for mode in ('raw_max', 'lower_max'): record.pop(mode, None)
                        write_json(path, record)
                        print(f'Invalid initialization; saved replacement {attempt}: {exc}', flush=True)
                    except (RuntimeError, ValueError) as exc:
                        record['error'] = str(exc)
                        print(f'Pair failed without replacement: {exc}', flush=True)
                        break
                write_json(path, record)
                records.append(record)
                write_json(args.outdir / 'summary.json', summarize(records))
                print(f"{record['status']} " + (' '.join(f"{m}:{record[m]['ending']}" for m in ('raw_max','lower_max')) if record['status']=='complete' else record['error']), flush=True)
        write_json(args.outdir / 'complete.json', summarize(records))
    finally:
        if env is not None: env.close()
        if world is not None:
            with suppress(RuntimeError):
                settings = world.get_settings(); settings.synchronous_mode = False; world.apply_settings(settings)
        if process is not None: stop(process)


if __name__ == '__main__': main()
