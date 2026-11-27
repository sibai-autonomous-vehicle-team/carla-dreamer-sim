"""Collect synchronized RGB braking episodes with a clean true-gap PID."""

import argparse
from contextlib import suppress
import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import subprocess
import threading
import time
from types import SimpleNamespace

import torch

from generate_dataset import save_atomic
from run_gap_pid import GapPID
from .closed_loop import read_snapshot
from .evaluate_closed_loop import make_scenarios, write_json
from .braking_scenario import BrakingScenario, InvalidBrakingScenario


CONDITIONS = [f'{day}_{weather}' for day in ('day', 'sunset', 'night')
              for weather in ('sunny', 'cloudy', 'rainy')]


class DatasetWriter:
    """Write the original per-condition tensor files, committing lengths last."""

    fields = ('actions', 'states', 'costs', 'distances')

    def __init__(self, root, conditions):
        self.root, self.data = Path(root), {}
        for condition in conditions:
            folder = self.root / condition
            (folder / 'obses').mkdir(parents=True, exist_ok=True)
            (folder / 'metadata').mkdir(exist_ok=True)
            lengths_path = folder / 'seq_lengths.pth'
            lengths = torch.load(lengths_path, weights_only=True) if lengths_path.exists() else torch.empty(0, dtype=torch.long)
            columns = {}
            for field in self.fields:
                path = folder / f'{field}.pth'
                values = torch.load(path, map_location='cpu', weights_only=True) if path.exists() else []
                if len(values) < len(lengths):
                    raise ValueError(f'Missing committed episodes in {path}')
                # A process can stop after a numerical file is replaced but before
                # lengths commit. Discard that uncommitted tail on resume.
                columns[field] = values[:len(lengths)]
                if len(values) != len(lengths):
                    save_atomic(columns[field], path)
            for index, length in enumerate(lengths.tolist()):
                if any(len(columns[field][index]) != length for field in self.fields):
                    raise ValueError(f'Invalid committed episode length in {folder}, episode {index}')
                for name in ('obses', 'metadata'):
                    if not (folder / name / f'episode_{index:03d}.pth').is_file():
                        raise ValueError(f'Missing committed {name} in {folder}, episode {index}')
            self.data[condition] = dict(columns=columns, lengths=lengths.tolist())

    def completed(self, condition):
        return len(self.data[condition]['lengths'])

    def summary(self, condition, index):
        path = self.root / condition / 'metadata' / f'episode_{index:03d}.pth'
        return torch.load(path, map_location='cpu', weights_only=True)['summary']

    def write(self, payload, condition, index):
        group = self.data[condition]
        if index != len(group['lengths']):
            raise ValueError('Episodes must be committed in condition-local order')
        length = len(payload['actions'])
        # Match generate_dataset.py exactly. Images/states precede each action,
        # while distance/collision readings describe its resulting transition.
        rows = dict(actions=payload['actions'].to(torch.float32),
                    states=payload['states'][:-1].to(torch.float32),
                    costs=payload['collision'].to(torch.float32),
                    distances=payload['distances'][1:].to(torch.float32))
        if length < 1 or len(payload['images']) != length + 1 or any(len(row) != length for row in rows.values()):
            raise ValueError('Episode tensors do not match the original dataset row contract')
        folder = self.root / condition
        filename = f'episode_{index:03d}.pth'
        save_atomic(payload['images'][:-1].to(torch.uint8).clone(), folder / 'obses' / filename)
        metadata = {key: value for key, value in payload.items()
                    if key not in ('images', 'states', 'actions', 'distances', 'collision')}
        metadata.update(initial_gap=payload['distances'][0].clone(),
                        terminal_image=payload['images'][-1].clone(),
                        terminal_state=payload['states'][-1].clone(),
                        condition_episode=index)
        save_atomic(metadata, folder / 'metadata' / filename)
        for field, row in rows.items():
            save_atomic(group['columns'][field] + [row], folder / f'{field}.pth')
        save_atomic(torch.tensor(group['lengths'] + [length], dtype=torch.long), folder / 'seq_lengths.pth')
        for field, row in rows.items():
            group['columns'][field].append(row)
        group['lengths'].append(length)


class BrowserMonitor:
    """Serve the existing CarDreamer viewer using bounded, latest-frame updates."""

    def __init__(self, port):
        from flask import Flask, Response, render_template
        from werkzeug.serving import make_server
        from car_dreamer.toolkit.monitor.monitor import EnvMonitorBase
        import car_dreamer.toolkit.monitor.monitor as monitor_module

        self.condition = threading.Condition()
        self.latest, self.version, self.closed = None, 0, False
        self.renderer = EnvMonitorBase.__new__(EnvMonitorBase)
        self.renderer._thread = None
        self.renderer._config = SimpleNamespace(display=SimpleNamespace(enable=True, render_keys=['visual']))
        app = Flask(__name__, template_folder=str(Path(monitor_module.__file__).parent / 'templates'))

        @app.route('/')
        def index():
            return render_template('index.html')

        @app.route('/stream')
        def stream():
            def frames():
                version = -1
                while True:
                    with self.condition:
                        self.condition.wait_for(lambda: self.closed or self.version != version, timeout=10)
                        if self.closed:
                            return
                        version, data = self.version, self.latest
                    if data is None:
                        yield ': waiting for collection\n\n'
                    else:
                        yield f'data: {data}\n\n'
            return Response(frames(), mimetype='text/event-stream',
                            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

        self.server = make_server('0.0.0.0', port, app, threaded=True)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def render(self, image, info):
        payload = json.dumps(self.renderer._render({'visual': image}, info), allow_nan=False)
        with self.condition:
            self.latest = payload
            self.version += 1
            self.condition.notify_all()

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def sample_plan(args, scenarios):
    rng = random.Random(args.seed)
    conditions = [condition for condition in args.conditions for _ in range(args.trajectories_per_condition)]
    rng.shuffle(conditions)
    plan = []
    for i, (scenario, condition) in enumerate(zip(scenarios, conditions)):
        gap = rng.uniform(*args.gap_range)
        plan.append(dict(episode=i, scenario=scenario, condition=condition,
                         initial_speed=rng.uniform(*args.speed_range), initial_gap=gap,
                         target_gap=gap if args.target_gap is None else args.target_gap,
                         brake_after=rng.uniform(*args.brake_delay_range)))
    if len(plan) != args.trajectories_per_condition * len(args.conditions):
        raise ValueError('One randomized world spawn is required per episode')
    return plan


def sample_state(env, dt):
    frame, timestamp, pose, gap = read_snapshot(env, dt)
    snapshot = env.world.get_snapshot()
    if snapshot.frame != frame:
        raise RuntimeError('World advanced during collection')
    velocities = []
    for actor in (env.ego, env.nonego):
        velocity = snapshot.find(actor.id).get_velocity()
        velocities.extend((velocity.x, velocity.y, velocity.z))
    return frame, timestamp, pose, gap, torch.tensor(velocities, dtype=torch.float32)


def collect_episode(env, plan, args, monitor):
    pid = GapPID(plan['target_gap'], args.kp, args.ki, args.kd)
    initial = sample_state(env, args.dt)
    if hasattr(args, 'gap_range') and not args.gap_range[0] <= initial[3] <= args.gap_range[1]:
        raise InvalidBrakingScenario('Measured starting bumper gap is outside the requested range')
    if hasattr(args, 'speed_range') and any(
            not args.speed_range[0] <= float(initial[4][start:start + 3].norm()) <= args.speed_range[1]
            for start in (0, 3)):
        raise InvalidBrakingScenario('Measured starting speed is outside the requested range')
    frames, timestamps, states, gaps, velocities, images = [], [], [], [], [], []
    actions, applied, leader_controls, transition_collision = [], [], [], []
    end_tick = math.ceil(plan['brake_after'] / args.dt) + math.ceil(args.brake_duration / args.dt)
    sample = initial

    def append_observation(value):
        frame, timestamp, pose, gap, velocity = value
        frames.append(frame)
        timestamps.append(timestamp)
        states.append(pose)
        gaps.append(gap)
        velocities.append(velocity)
        images.append(env.get_rgb_observation(frame).clone())

    append_observation(sample)
    ending = 'braking_complete'
    for tick in range(end_tick):
        env.validate_lanes()
        action = env.nominal_action(pid)
        braking = tick * args.dt >= plan['brake_after']
        if monitor:
            monitor.render(images[-1].numpy(), dict(
                episode=f"{plan['episode'] + 1}/{args.n_traj}", condition=plan['condition'],
                phase='leader braking' if braking else 'following',
                gap_m=gaps[-1], target_gap_m=plan['target_gap'],
                ego_speed_mps=float(velocities[-1][:3].norm()),
                leader_speed_mps=float(velocities[-1][3:].norm()),
                elapsed_s=tick * args.dt, brake_after_s=plan['brake_after'],
                longitudinal=action[0], leader_brake=float(braking)))
        env.step(action)
        sample = sample_state(env, args.dt)
        if sample[0] != frames[-1] + 1 or not math.isclose(sample[1] - timestamps[-1], args.dt, abs_tol=1e-4):
            raise RuntimeError('Nonconsecutive collection frames')
        control = env.ego.get_control()
        leader = env.nonego.get_control()
        actions.append(action)
        applied.append((control.throttle, control.brake, control.steer))
        leader_controls.append((leader.throttle, leader.brake, leader.steer))
        append_observation(sample)
        # Contact sensors are asynchronous. Drain each tick so a collision ends
        # the trajectory before another control command is issued.
        time.sleep(args.collision_grace)
        collision = any(initial[0] <= frame <= sample[0] for frame, _ in env.events)
        transition_collision.append(collision)
        if not collision:
            env.validate_lanes()
        if any(other != env.ego.id for _, other in env.leader_events):
            raise InvalidBrakingScenario('Leader contacted an object outside the following pair')
        if collision:
            ending = 'collision'
            break
    measurements = env.measurements()
    events = [(frame, other) for frame, other in measurements['collision_events']
              if frames[0] <= frame <= frames[-1]]
    if events:
        first = min(frame for frame, _ in events)
        # Late delivery still cuts the saved episode at the collision snapshot.
        count = next(i for i, frame in enumerate(frames[1:], 1) if frame >= first)
        frames, timestamps, states, gaps, velocities, images = [
            values[:count + 1] for values in (frames, timestamps, states, gaps, velocities, images)]
        actions, applied, leader_controls = [values[:count] for values in (actions, applied, leader_controls)]
        transition_collision = [False] * (count - 1) + [True]
        ending = 'collision'
    if any(other != env.nonego.id for _, other in events):
        raise InvalidBrakingScenario('Ego contacted an object outside the following pair')
    if any(other != env.ego.id for _, other in env.leader_events):
        raise InvalidBrakingScenario('Leader contacted an object outside the following pair')
    payload = dict(format_version=2, plan=plan, dt=args.dt,
                   frames=torch.tensor(frames), timestamps=torch.tensor(timestamps, dtype=torch.float64),
                   images=torch.stack(images), states=torch.stack(states), distances=torch.tensor(gaps),
                   velocities=torch.stack(velocities), actions=torch.tensor(actions, dtype=torch.float32),
                   applied_controls=torch.tensor(applied, dtype=torch.float32),
                   leader_controls=torch.tensor(leader_controls, dtype=torch.float32),
                   collision=torch.tensor(transition_collision, dtype=torch.bool), ending=ending,
                   collision_events=events, ego_id=env.ego.id, leader_id=env.nonego.id,
                   lane_departure_frames=measurements['lane_departure_frames'])
    summary = dict(episode=plan['episode'], condition=plan['condition'], ending=ending,
                   transitions=len(actions), initial_gap_m=gaps[0], target_gap_m=plan['target_gap'],
                   requested_speed_mps=plan['initial_speed'], ego_speed_mps=float(velocities[0][:3].norm()),
                   leader_speed_mps=float(velocities[0][3:].norm()), min_gap_m=min(gaps), max_gap_m=max(gaps),
                   first_frame=frames[0], last_frame=frames[-1], collision_events=events,
                   leader_collision=any(other == env.nonego.id for _, other in events),
                   collision_before_braking=bool(events) and not any(row[1] for row in leader_controls))
    if monitor:
        monitor.render(images[-1].numpy(), dict(
            episode=f"{plan['episode'] + 1}/{args.n_traj}", condition=plan['condition'], phase=ending,
            gap_m=gaps[-1], target_gap_m=plan['target_gap'], elapsed_s=len(actions) * args.dt,
            ego_speed_mps=float(velocities[-1][:3].norm()), leader_speed_mps=float(velocities[-1][3:].norm())))
    return payload, summary


def server_environment(source=None):
    """Respect explicit Vulkan selection; otherwise choose an existing NVIDIA ICD.

    RIS uses nvidia_icd.x86_64.json, while local Ubuntu commonly uses
    nvidia_icd.json. Never force the loader to read a nonexistent manifest.
    """
    environment=dict(os.environ if source is None else source)
    if not environment.get('VK_ICD_FILENAMES') and not environment.get('VK_DRIVER_FILES'):
        for directory in ('/usr/share/vulkan/icd.d','/etc/vulkan/icd.d'):
            candidates=sorted(Path(directory).glob('nvidia_icd*.json'))
            if candidates:
                environment['VK_ICD_FILENAMES']=str(candidates[0])
                break
    return environment


def start_server(args):
    import carla

    for port in (args.port, args.port + 1, args.port + 2):
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('0.0.0.0', port))
    log = (args.outdir / 'server.log').open('w')
    environment = server_environment()
    startup_map=(['/Game/Carla/Maps/OpenDriveMap']
                 if getattr(args,'road_environment','town')=='straight' else [])
    process = subprocess.Popen(
        [str(args.server.resolve()), *startup_map, '-RenderOffScreen', '-ResX=256', '-ResY=256',
         '-quality-level=Epic', '-nosound', f'-carla-rpc-port={args.port}'],
        stdout=log, stderr=subprocess.STDOUT, env=environment, start_new_session=True)
    log.close()
    from collect_parallel import stop
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f'CARLA exited with code {process.returncode}. Inspect {args.outdir / "server.log"}. '
                                   f'Vulkan driver selection: {environment.get("VK_DRIVER_FILES", environment.get("VK_ICD_FILENAMES", "automatic"))}')
            client = carla.Client('127.0.0.1', args.port)
            client.set_timeout(2.)
            try:
                client.get_server_version()
                return process
            except RuntimeError:
                time.sleep(1.)
        raise RuntimeError('CARLA startup timed out')
    except BaseException:
        stop(process)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--trajectories-per-condition', '--n-traj', type=int, default=500,
                        help='Accepted episodes for each selected condition (default 500, total 4500 for all nine)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--speed-range', type=float, nargs=2, default=[12., 20.])
    parser.add_argument('--gap-range', type=float, nargs=2, default=[2., 5.])
    parser.add_argument('--target-gap', type=float, default=3., help='Desired bumper gap in metres (default: 3)')
    parser.add_argument('--brake-delay-range', type=float, nargs=2, default=[1.5, 2.5])
    parser.add_argument('--brake-duration', type=float, default=2.)
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=CONDITIONS)
    parser.add_argument('--kp', type=float, default=.9)
    parser.add_argument('--ki', type=float, default=.05)
    parser.add_argument('--kd', type=float, default=1.)
    parser.add_argument('--dt', type=float, default=.1)
    parser.add_argument('--port', type=int, default=2700)
    parser.add_argument('--start-server', action='store_true')
    parser.add_argument('--server', type=Path, default=Path('.local/carla-0.9.15/CarlaUE4.sh'))
    parser.add_argument('--headless', action='store_true', help='Disable the existing CarDreamer browser viewer')
    parser.add_argument('--real-time', action='store_true')
    parser.add_argument('--collision-grace', type=float, default=.02)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    for name in ('speed_range', 'gap_range', 'brake_delay_range'):
        low, high = getattr(args, name)
        if not all(map(math.isfinite, (low, high))) or not 0 < low <= high:
            parser.error(f'--{name.replace("_", "-")} requires positive ordered bounds')
    args.n_traj = args.trajectories_per_condition * len(args.conditions)
    if len(set(args.conditions)) != len(args.conditions):
        parser.error('--conditions must be unique')
    if args.n_traj < 1 or args.dt <= 0 or args.brake_duration <= 0 or args.collision_grace < 0:
        parser.error('Episode count, timestep and braking duration must be positive. Collision grace must be nonnegative')
    if not all(map(math.isfinite, (args.dt, args.brake_duration, args.collision_grace, args.kp, args.ki, args.kd))):
        parser.error('Collection settings must be finite')
    if args.target_gap is not None and (not math.isfinite(args.target_gap) or args.target_gap <= 0):
        parser.error('--target-gap must be positive and finite')
    if args.start_server and not args.server.is_file():
        parser.error('CARLA server launcher not found')
    if args.resume and not (args.outdir / 'manifest.json').is_file():
        parser.error('--resume requires an existing collection manifest')
    if args.outdir.exists() and not args.resume:
        parser.error('Output already exists. Use a fresh directory or --resume')
    args.outdir.mkdir(parents=True, exist_ok=args.resume)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                if k not in ('resume', 'headless', 'real_time', 'start_server')}
    manifest = dict(format_version=3, dataset_layout='original_condition_files',
                    lane_controller='persisted-route pure pursuit for both vehicles, including initialization',
                    settings=settings, controller='true bumper-gap PID, zero action noise',
                    gap_definition='CarlaFollowEnv.get_distance front/rear bounding-box face-edge distance',
                    observations='T pre-action RGB/state/action rows, T post-action distance/collision rows, matching generate_dataset.py',
                    extra_metadata='condition/metadata/episode_XXX.pth stores frame IDs, timestamps, velocities, controls, terminal observation and summary',
                    action_convention='requested longitudinal [-3,3], requested steering [-1,1]; applied steer has opposite sign',
                    control_columns=['throttle', 'brake', 'applied_steer'],
                    stopping_rule='first collision snapshot or end of scheduled full leader braking',
                    browser_url=f'http://localhost:{args.port + 7000}',
                    code_sha256=hashlib.sha256(Path(__file__).read_bytes() +
                                               Path(__file__).with_name('braking_scenario.py').read_bytes() +
                                               Path(__file__).with_name('evaluate_closed_loop.py').read_bytes()).hexdigest())
    if args.resume:
        previous = json.loads((args.outdir / 'manifest.json').read_text())
        if previous.get('dataset_layout') != manifest['dataset_layout']:
            parser.error('This output uses an earlier dataset layout. Choose a fresh output directory')
        if previous['settings'] != settings:
            parser.error('Resume settings differ from this dataset')
        hashes = previous.setdefault('resume_code_sha256', [])
        if manifest['code_sha256'] not in hashes:
            hashes.append(manifest['code_sha256'])
        write_json(args.outdir / 'manifest.json', previous)
    else:
        write_json(args.outdir / 'manifest.json', manifest)
    writer = DatasetWriter(args.outdir, args.conditions)
    import carla
    from collect_parallel import stop

    process, monitor, env, world = None, None, None, None
    try:
        if not args.headless:
            monitor = BrowserMonitor(args.port + 7000)
            print(f'Browser viewer: {manifest["browser_url"]}', flush=True)
        if args.start_server:
            process = start_server(args)
        client = carla.Client('127.0.0.1', args.port)
        client.set_timeout(120.)
        world = client.load_world('Town03')
        plan_path = args.outdir / 'plan.json'
        steps = math.ceil((max(args.brake_delay_range) + args.brake_duration) / args.dt)
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
        else:
            scenarios = []
            while len(scenarios) < args.n_traj:
                count = min(500, args.n_traj - len(scenarios))
                scenarios.extend(make_scenarios(world, count, args.seed + len(scenarios), steps))
            plan = sample_plan(args, scenarios)
            write_json(plan_path, plan)
        condition_indices = dict.fromkeys(args.conditions, 0)
        for entry in plan:
            condition = entry['condition']
            index = condition_indices[condition]
            condition_indices[condition] += 1
            if index < writer.completed(condition):
                continue
            rollout = SimpleNamespace(condition=entry['condition'], blueprint='vehicle.audi.a2',
                                      initial_gap=entry['initial_gap'], initial_speed=entry['initial_speed'],
                                      leader_brake_test=True, leader_brake_after=entry['brake_after'],
                                      leader_brake_duration=args.brake_duration, stationary_leader_test=False,
                                      graphics='vulkan', camera=True, real_time=args.real_time,
                                      handoff_gap_bounds=args.gap_range, handoff_speed_bounds=args.speed_range)
            print(f"Episode {entry['episode'] + 1}/{args.n_traj}: {entry['condition']}, "
                  f"speed={entry['initial_speed']:.2f} m/s, gap={entry['initial_gap']:.2f} m, "
                  f"brake at {entry['brake_after']:.2f} s", flush=True)
            attempt, simulator_failures = 0, 0
            while True:
                attempt += 1
                try:
                    env = BrakingScenario(client, rollout, args.dt, entry['scenario'], reload_world=False)
                    payload, summary = collect_episode(env, entry, args, monitor)
                    payload['summary'] = summary
                    payload['collector_sha256'] = manifest['code_sha256']
                    break
                except (RuntimeError, ValueError) as error:
                    invalid_scenario = isinstance(error, InvalidBrakingScenario)
                    with (args.outdir / 'failures.jsonl').open('a') as file:
                        file.write(json.dumps(dict(episode=entry['episode'], attempt=attempt, error=str(error),
                                                   invalid_scenario=invalid_scenario)) + '\n')
                    print(f"Episode {entry['episode'] + 1} attempt {attempt} failed: {error}", flush=True)
                    simulator_failures = 0 if invalid_scenario else simulator_failures + 1
                    if simulator_failures >= 30:
                        raise
                    if invalid_scenario or simulator_failures % 3 == 0:
                        replacements = entry.setdefault('replaced_spawns', [])
                        replacement_seed = args.seed + 1000003 * (entry['episode'] + 1) + 104729 * len(replacements)
                        replacement = make_scenarios(world, 1, replacement_seed, steps)[0]
                        replacements.append(entry['scenario'])
                        entry['scenario'] = replacement
                        write_json(plan_path, plan)
                        print('Retrying this episode with another random road spawn', flush=True)
                finally:
                    if env is not None:
                        env.close()
                        env = None
            writer.write(payload, condition, index)
            print(f"Saved {condition}/obses/episode_{index:03d}.pth: "
                  f"{summary['ending']}, {summary['transitions']} transitions", flush=True)
        summaries = [writer.summary(condition, index) for condition in args.conditions
                     for index in range(writer.completed(condition))]
        summaries.sort(key=lambda row: row['episode'])
        write_json(args.outdir / 'summary.json', summaries)
        write_json(args.outdir / 'complete.json', dict(episodes=len(summaries),
                   collisions=sum(row['ending'] == 'collision' for row in summaries),
                   episodes_per_condition={condition: sum(row['condition'] == condition for row in summaries)
                                           for condition in args.conditions}))
    finally:
        if env is not None:
            env.close()
        if world is not None:
            with suppress(RuntimeError):
                current = client.get_world()
                settings = current.get_settings()
                settings.synchronous_mode = False
                current.apply_settings(settings)
        if process is not None:
            stop(process)
        if monitor is not None:
            monitor.close()


if __name__ == '__main__':
    main()
