"""Matched, frozen-policy CARLA evaluation on a dedicated server.

Example: python -m training.evaluate_closed_loop --start-server --checkpoints PATH
This loads Town03 and reloads the world between rollouts. Never share the server
with a collector or another ticking client. RGB policies and nominal gap CNNs
use synchronized camera observations.
"""
import argparse
import csv
from contextlib import suppress
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import random
import socket
import subprocess
import time
from types import SimpleNamespace

from .closed_loop import read_snapshot, run_matched_scenario
from .models import SafetyFilter

BRAKE_TEST_SPEED_MPS = 2.0
BRAKE_TEST_START_S = 0.5
BRAKE_TEST_DURATION_S = 1.0


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


class NominalPID:
    """Propose PID actions from the selected gap source at the decision frame."""

    def __init__(self, pid, dt, estimate=None):
        self.pid, self.dt, self.estimate = pid, dt, estimate
        self.gap_m = None
        self.image_frame = None

    def __call__(self, env):
        from run_gap_pid import steering_command

        if self.estimate is None:
            self.gap_m = float(env.get_distance())
        else:
            frame = env.ego.get_world().get_snapshot().frame
            image = env.get_rgb_observation(frame)
            self.gap_m = float(self.estimate(image.numpy()))
            self.image_frame = frame
        if not math.isfinite(self.gap_m):
            raise ValueError('Nonfinite nominal gap estimate')
        return self.pid.update(self.gap_m, self.dt), steering_command(env)


class ScoreDisplay:
    """Show a chase-camera frame with the decision-time nominal HJ score."""

    def __init__(self):
        import pygame

        self.pygame = pygame
        pygame.display.init()
        pygame.font.init()
        self.window = pygame.display.set_mode((1280, 720))
        pygame.display.set_caption('HJ evaluation')
        self.font = pygame.font.Font(None, 32)
        self.override_label = 'HJ override'

    def show(self, image, score, mode, intervened, status_override=None):
        pygame = self.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                raise KeyboardInterrupt('Evaluation window closed')
        self.window.blit(pygame.surfarray.make_surface(image.swapaxes(0, 1)), (0, 0))
        value = 'unavailable' if score is None else f'{score:+.3f} m'
        status = self.override_label if intervened else {'pid': 'PID', 'safety_filter': 'PID', 'actor_only': 'Actor'}[mode]
        if status_override is not None:
            status = status_override
        color = (255, 165, 100) if intervened else (240, 245, 250)
        labels = [self.font.render(f'HJ Q(PID)  {value}', True, color),
                  self.font.render(status, True, color)]
        width = max(label.get_width() for label in labels) + 24
        height = sum(label.get_height() for label in labels) + 30
        panel = pygame.Surface((width, height), pygame.SRCALPHA)
        panel.fill((12, 18, 26, 225))
        left = self.window.get_width() - width - 16
        self.window.blit(panel, (left, 16))
        y = 28
        for label in labels:
            self.window.blit(label, (self.window.get_width() - label.get_width() - 28, y))
            y += label.get_height() + 6
        pygame.display.flip()

    def close(self):
        self.pygame.display.quit()
        self.pygame.font.quit()


def pose_values(transform):
    p, r = transform.location, transform.rotation
    return [p.x, p.y, p.z, r.pitch, r.yaw, r.roll]


def transform_from(values):
    import carla
    return carla.Transform(carla.Location(*values[:3]),
                           carla.Rotation(pitch=values[3], yaw=values[4], roll=values[5]))


def make_scenarios(world, count, seed, steps):
    """Persist a deterministic lane-following leader route, including branch choices."""
    waypoints = sorted(world.get_map().generate_waypoints(5.),
                       key=lambda w: (w.road_id, w.section_id, w.lane_id, w.s))
    rng = random.Random(seed)
    rng.shuffle(waypoints)
    scenarios = []
    for start in waypoints:
        if start.is_junction:
            continue
        route = [start]
        for _ in range(max(120, int(steps * 1.5))):
            successors = sorted(route[-1].next(2.), key=lambda w: (w.road_id, w.lane_id, w.s))
            if not successors:
                break
            route.append(successors[0])
        if len(route) < max(40, int(steps*.2)):
            continue
        # Ten metres center-to-center, as in the collection reset.
        ego, leader = pose_values(start.transform), pose_values(route[5].transform)
        ego[2] += .5
        leader[2] += .5
        scenarios.append(dict(id=f'scenario_{len(scenarios):03d}', seed=seed+len(scenarios),
                              ego=ego, leader=leader, route=[pose_values(w.transform) for w in route[5:]],
                              full_route=[pose_values(w.transform) for w in route]))
        if len(scenarios) == count:
            return scenarios
    raise ValueError('Not enough suitable map routes for the requested scenarios')


class LiveScenario:
    """Minimal synchronous adapter for the existing closed-loop helpers."""
    def __init__(self, client, args, dt, scenario, *, reload_world=True):
        import carla
        from generate_dataset import make_weather
        self.carla, self.args, self.dt = carla, args, dt
        self.actors, self.events, self.departures = [], [], []
        self.imu_frames = queue.Queue()
        self.rgb_frames = queue.Queue()
        self.current_rgb = None
        self.current_rgb_frame = None
        self.chase_frames = queue.Queue()
        self.chase_camera = None
        self.current_chase = None
        self.display = None
        self.wants_hj_score = args.graphics == 'windowed'
        self._config = SimpleNamespace(action=SimpleNamespace(discrete=False))
        self.world = client.reload_world(False) if reload_world else client.get_world()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = dt
        settings.substepping = True
        settings.max_substep_delta_time = .01
        settings.max_substeps = math.ceil(dt/.01)
        settings.no_rendering_mode = args.graphics != 'windowed' and not args.camera
        self.world.apply_settings(settings)
        tod, weather = args.condition.split('_')
        parameters=getattr(args,'weather_parameters',None)
        self.world.set_weather(parameters if parameters is not None else make_weather(tod,weather))
        self.map = self.world.get_map()
        self.route, self.route_index = scenario['route'], 0
        try:
            bp = self.world.get_blueprint_library().find(args.blueprint)
            bp.set_attribute('role_name', 'evaluation')
            for key in ('ego', 'leader'):
                actor = self.world.spawn_actor(bp, transform_from(scenario[key]))
                self.actors.append(actor)
                actor.apply_control(carla.VehicleControl(brake=1.))
                if getattr(args,'vehicle_initial_lights',None) is not None:
                    actor.set_light_state(args.vehicle_initial_lights)
            self.ego, self.nonego = self.actors
            for _ in range(20):
                self.world.tick()
            if args.initial_gap is not None:
                self.set_initial_gap(args.initial_gap)
            if args.stationary_leader_test:
                self.ego.apply_control(carla.VehicleControl(throttle=1.))
                for _ in range(math.ceil(5 / self.dt)):
                    self.world.tick()
                    if self.get_distance() <= 4.:
                        break
                else:
                    raise RuntimeError('Ego did not reach the stationary-leader approach gap')
            if args.leader_brake_test and args.initial_speed is None:
                for actor in (self.ego, self.nonego):
                    actor.apply_control(carla.VehicleControl(throttle=2/3))
                for _ in range(math.ceil(5 / self.dt)):
                    self.world.tick()
                    if all(actor.get_velocity().length() >= BRAKE_TEST_SPEED_MPS
                           for actor in (self.ego, self.nonego)):
                        break
                else:
                    raise RuntimeError('Vehicles did not reach the braking test start speed')
            if args.initial_speed is not None:
                initial_speed = args.initial_speed
                bounds = getattr(args, 'handoff_speed_bounds', None)
                if bounds is not None and bounds[1] - bounds[0] > .4:
                    initial_speed = max(bounds[0] + .15, min(bounds[1] - .15, initial_speed))
                self.initialize_speed(initial_speed)
            library = self.world.get_blueprint_library()
            sensor = self.world.spawn_actor(library.find('sensor.other.collision'), carla.Transform(), attach_to=self.ego)
            self.actors.append(sensor)
            sensor.listen(lambda e: self.events.append((e.frame, e.other_actor.id)))
            # An every-tick sensor provides a frame barrier without camera latency.
            sensor = self.world.spawn_actor(library.find('sensor.other.imu'), carla.Transform(), attach_to=self.ego)
            self.actors.append(sensor)
            sensor.listen(lambda e: self.imu_frames.put(e.frame))
            if args.camera:
                # Exact collection camera configuration: RGB 224x224, fov 120,
                # attached at x=1.5 m, z=2 m, one image per simulation tick.
                bp = library.find('sensor.camera.rgb')
                attributes={'image_size_x':'224', 'image_size_y':'224',
                            'fov':'120', 'sensor_tick':'0.0'}
                attributes.update(getattr(args,'camera_attributes',{}))
                for name,value in attributes.items():
                    bp.set_attribute(name,value)
                sensor = self.world.spawn_actor(bp,carla.Transform(carla.Location(x=1.5,z=2.)),attach_to=self.ego)
                self.actors.append(sensor)
                def rgb_callback(event):
                    import numpy as np
                    bgra = np.frombuffer(event.raw_data,dtype=np.uint8).reshape(event.height,event.width,4)
                    self.rgb_frames.put((event.frame,torch.from_numpy(bgra[:,:,:3][:,:,::-1].copy())))
                import torch
                sensor.listen(rgb_callback)
            if self.wants_hj_score:
                import numpy as np

                self.display = ScoreDisplay()
                self.follow_ego()
                bp = library.find('sensor.camera.rgb')
                for name, value in {'image_size_x': '1280', 'image_size_y': '720',
                                    'fov': '90', 'sensor_tick': '0.0'}.items():
                    bp.set_attribute(name, value)
                self.chase_camera = self.world.spawn_actor(bp, self.world.get_spectator().get_transform())
                self.actors.append(self.chase_camera)
                def chase_callback(event):
                    bgra = np.frombuffer(event.raw_data, dtype=np.uint8).reshape(event.height, event.width, 4)
                    self.chase_frames.put((event.frame, bgra[:, :, :3][:, :, ::-1].copy()))
                self.chase_camera.listen(chase_callback)
            self.tick()
            if getattr(args, 'handoff_gap_bounds', None) is not None:
                self.set_moving_gap(args.handoff_gap_bounds)
            self.validate_handoff_speed()
            self.initial_frame = self.world.get_snapshot().frame
            self.was_outside = self.outside_lane()
            if self.was_outside:
                self.departures.append(self.initial_frame)
        except BaseException:
            self.close()
            raise

    def validate_handoff_speed(self):
        if self.args.initial_speed is not None and any(
                abs(actor.get_velocity().length() - self.args.initial_speed) > .2
                for actor in (self.ego, self.nonego)):
            raise RuntimeError('Measured handoff speed differs from requested initial speed by more than 0.2 m/s')

    def set_initial_gap(self, gap_target):
        # Adjust along the spawn heading using the same gap metric as the trace.
        carla = self.carla
        ego_pose = self.ego.get_transform()
        leader_pose = self.nonego.get_transform()
        forward = ego_pose.get_forward_vector()
        for _ in range(5):
            error = self.get_distance(ego_pose, leader_pose) - gap_target
            if abs(error) <= .01:
                break
            location = ego_pose.location
            ego_pose.location = carla.Location(location.x + error * forward.x,
                                               location.y + error * forward.y,
                                               location.z + error * forward.z)
        self.ego.set_transform(ego_pose)
        for _ in range(20):
            self.world.tick()
        _, _, _, gap = read_snapshot(self, self.dt)
        # Teleporting on graded roads can shift suspension height while
        # the car settles. Correct against the resulting bumper faces.
        for _ in range(3):
            if abs(gap - gap_target) <= .05:
                break
            ego_pose = self.ego.get_transform()
            forward = ego_pose.get_forward_vector()
            error = gap - gap_target
            ego_pose.location += carla.Location(x=error * forward.x, y=error * forward.y,
                                               z=error * forward.z)
            self.ego.set_transform(ego_pose)
            for _ in range(20):
                self.world.tick()
            _, _, _, gap = read_snapshot(self, self.dt)
        if abs(gap - gap_target) > .05:
            raise ValueError(f'Settled gap {gap:.3f} m differs from requested {gap_target:.3f} m')

    def set_moving_gap(self, bounds):
        """Correct warmup displacement and capture a fresh aligned camera frame."""
        low, high = bounds
        target = (low + high) / 2
        for _ in range(5):
            measured = self.get_distance()
            if low <= measured <= high:
                if self.events:
                    raise RuntimeError('Contact occurred during moving-start initialization')
                return
            transform = self.ego.get_transform()
            leader = self.nonego.get_location()
            dx, dy = leader.x - transform.location.x, leader.y - transform.location.y
            separation = math.hypot(dx, dy)
            if separation < 1e-6:
                break
            error = measured - target
            velocity = self.ego.get_velocity()
            transform.location += self.carla.Location(x=error * dx / separation, y=error * dy / separation)
            self.ego.set_transform(transform)
            self.ego.set_target_velocity(velocity)
            self.tick()
        raise RuntimeError('Could not establish the requested moving bumper gap')

    def initialize_speed(self, speed):
        """Settle wheel dynamics before handing off an already-moving pair."""
        actors = (self.ego, self.nonego)
        commands = [speed, speed]
        for actor in actors:
            actor.apply_control(self.carla.VehicleControl())
        for _ in range(math.ceil(3 / self.dt)):
            for actor, command in zip(actors, commands):
                forward = actor.get_transform().get_forward_vector()
                actor.set_target_velocity(self.carla.Vector3D(
                    x=command * forward.x, y=command * forward.y, z=command * forward.z))
            self.world.tick()
            measured = [actor.get_velocity().length() for actor in actors]
            if all(abs(value - speed) <= .05 for value in measured):
                return
            commands = [max(0., min(speed + 3., command + .5 * (speed - value)))
                        for command, value in zip(commands, measured)]
        raise RuntimeError(f'Vehicles did not settle at {speed:.3f} m/s; measured speeds {measured}')

    def follow_ego(self):
        tf = self.ego.get_transform()
        forward = tf.get_forward_vector()
        c = self.carla
        camera = c.Transform(c.Location(x=tf.location.x-8*forward.x,
                                        y=tf.location.y-8*forward.y, z=tf.location.z+5),
                             c.Rotation(pitch=-22, yaw=tf.rotation.yaw))
        self.world.get_spectator().set_transform(camera)
        if self.chase_camera is not None:
            self.chase_camera.set_transform(camera)

    def tick(self):
        if self.args.graphics == 'windowed':
            self.follow_ego()
        frame = self.world.tick()
        deadline = time.monotonic()+10
        received = None
        while True:
            try:
                received = self.imu_frames.get(timeout=max(.001, deadline-time.monotonic()))
            except queue.Empty as exc:
                raise RuntimeError(f'IMU timeout waiting for frame {frame}; last IMU frame {received}') from exc
            if received == frame:
                break
            if received > frame or time.monotonic() > deadline:
                raise RuntimeError('Unusable sensor-frame sequence')
        if self.args.camera:
            deadline = time.monotonic()+30
            while True:
                try:
                    rgb_frame,image = self.rgb_frames.get(timeout=max(.001,deadline-time.monotonic()))
                except queue.Empty as exc:
                    raise RuntimeError(f'Camera timeout waiting for frame {frame}; last RGB frame {self.current_rgb_frame}') from exc
                if rgb_frame == frame:
                    self.current_rgb_frame,self.current_rgb = rgb_frame,image
                    break
                if rgb_frame > frame or time.monotonic() > deadline:
                    raise RuntimeError('Camera frame does not match the action-time snapshot')
        if self.display is not None:
            deadline = time.monotonic() + 30
            while True:
                try:
                    chase_frame, image = self.chase_frames.get(timeout=max(.001, deadline - time.monotonic()))
                except queue.Empty as exc:
                    raise RuntimeError(f'Chase-camera timeout waiting for frame {frame}') from exc
                if chase_frame == frame:
                    self.current_chase = image
                    break
                if chase_frame > frame or time.monotonic() > deadline:
                    raise RuntimeError('Chase-camera frame does not match the decision snapshot')
        return frame

    def render_decision(self, score, mode, intervened):
        if self.display is not None:
            self.display.show(self.current_chase, score, mode, intervened)

    def get_rgb_observation(self,frame):
        if self.current_rgb_frame != frame or self.current_rgb is None:
            raise RuntimeError('No synchronized RGB observation for this decision frame')
        return self.current_rgb

    def get_distance(self, ego_transform=None, leader_transform=None):
        from car_dreamer.carla_follow_env import CarlaFollowEnv
        return CarlaFollowEnv.get_distance(self, ego_transform, leader_transform)

    def outside_lane(self):
        # Explicit metric: ego center outside any driving lane (not lane-marking crossings).
        c = self.carla
        wp = self.map.get_waypoint(self.ego.get_location(), project_to_road=False, lane_type=c.LaneType.Driving)
        return wp is None

    def leader_control(self):
        # Collection-style leader: throttle 2/3 below 2 m/s, heading feedback.
        c = self.carla
        if self.args.stationary_leader_test:
            return c.VehicleControl(brake=1.)
        tf, velocity = self.nonego.get_transform(), self.nonego.get_velocity()
        distances = [math.hypot(p[0]-tf.location.x, p[1]-tf.location.y) for p in self.route[self.route_index:]]
        self.route_index += min(range(len(distances)), key=distances.__getitem__)
        target = self.route[min(self.route_index+2, len(self.route)-1)]
        heading = math.degrees(math.atan2(target[1]-tf.location.y, target[0]-tf.location.x))
        error = (heading-tf.rotation.yaw+180)%360-180
        steer = max(-1., min(1., .03*error))
        if self.args.leader_brake_test:
            elapsed = (self.world.get_snapshot().frame - self.initial_frame) * self.dt
            if self.args.leader_brake_after <= elapsed < self.args.leader_brake_after + self.args.leader_brake_duration:
                return c.VehicleControl(brake=1., steer=steer)
        target_speed = self.args.initial_speed if self.args.initial_speed is not None else 2.
        return c.VehicleControl(throttle=2/3 if math.hypot(velocity.x, velocity.y)<target_speed else 0., steer=steer)

    def step(self, action):
        from car_dreamer.carla_base_env import CarlaBaseEnv
        started = time.monotonic()
        self.ego.apply_control(CarlaBaseEnv.get_vehicle_control(self, action))
        self.nonego.apply_control(self.leader_control())
        frame = self.tick()
        outside = self.outside_lane()
        if outside and not self.was_outside:
            self.departures.append(frame)
        self.was_outside = outside
        if (self.args.graphics == 'windowed' or self.args.real_time) and not getattr(self.args, 'external_pacing', False):
            time.sleep(max(0., self.dt-(time.monotonic()-started)))
        # Run a fixed horizon, including post-violation behavior. Sensor contacts
        # are summarized later rather than relying on asynchronous done flags.
        return None, 0., False, {}

    def measurements(self):
        # Event-only sensors have no per-frame watermark. Allow callback delivery
        # after the final IMU barrier; preserve this limitation in the manifest.
        time.sleep(.2)
        return dict(collision_events=list(self.events), lane_departure_frames=list(self.departures))

    def close(self):
        for actor in reversed(self.actors):
            with suppress(RuntimeError):
                if hasattr(actor, 'stop'):
                    actor.stop()
                actor.destroy()
        self.actors = []
        if getattr(self, 'display', None) is not None:
            self.display.close()
            self.display = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--condition', default='day_sunny', choices=[f'{t}_{w}' for t in ('day','sunset','night') for w in ('sunny','cloudy','rainy')])
    parser.add_argument('--target-gap', type=float, default=5.)
    parser.add_argument('--nominal-cnn', type=Path,
                        help='Frozen gap CNN checkpoint for nominal PID control')
    parser.add_argument('--initial-gap', type=float, help='Requested measured gap at rollout start in meters')
    parser.add_argument('--initial-speed', type=float,
                        help='Initialize both vehicles already moving at this speed in m/s')
    parser.add_argument('--leader-brake-test', action='store_true',
                        help='Start both cars moving and apply a brief full-brake pulse to the leader')
    parser.add_argument('--leader-brake-after', type=float, default=BRAKE_TEST_START_S,
                        help='Seconds after control handoff before the leader applies full brake')
    parser.add_argument('--leader-brake-duration', type=float, default=BRAKE_TEST_DURATION_S,
                        help='Duration of the leader full-brake pulse in seconds')
    parser.add_argument('--stationary-leader-test', action='store_true',
                        help='Accelerate ego toward a parked leader before handing off to the policy')
    parser.add_argument('--kp', type=float, default=.5)
    parser.add_argument('--ki', type=float, default=.25,
                        help='Integral gain for eliminating steady-state gap error (default: .25)')
    parser.add_argument('--kd', type=float, default=.5)
    parser.add_argument('--blueprint', default='vehicle.audi.a2')
    parser.add_argument('--port', type=int, default=2600)
    parser.add_argument('--start-server', action='store_true')
    parser.add_argument('--graphics', choices=('null', 'vulkan', 'windowed'), default=None,
                        help='null: no graphics; vulkan: off-screen; windowed: visible chase camera at real-time speed')
    parser.add_argument('--real-time', action='store_true', help='Pace each action to the simulation timestep')
    parser.add_argument('--modes', nargs='+', choices=('pid', 'actor_only', 'safety_filter'),
                        default=['pid', 'actor_only', 'safety_filter'], help='Run selected modes sequentially')
    parser.add_argument('--server', type=Path, default=Path('.local/carla-0.9.15/CarlaUE4.sh'))
    args = parser.parse_args()
    if args.stationary_leader_test and (args.leader_brake_test or args.initial_gap is None or args.initial_gap <= 4):
        parser.error('--stationary-leader-test requires --initial-gap above 4 m and no leader brake pulse')
    if args.initial_gap is not None and (not math.isfinite(args.initial_gap) or args.initial_gap <= 0):
        parser.error('--initial-gap must be finite and positive')
    if args.initial_speed is not None and (not math.isfinite(args.initial_speed) or args.initial_speed <= 0
                                           or args.stationary_leader_test):
        parser.error('--initial-speed requires a positive finite speed and a moving leader')
    if (not math.isfinite(args.leader_brake_after) or args.leader_brake_after < 0
            or not math.isfinite(args.leader_brake_duration) or args.leader_brake_duration <= 0):
        parser.error('Leader brake delay must be nonnegative and duration must be positive and finite')
    if min(args.episodes, args.steps) < 1 or not math.isfinite(args.target_gap) or args.target_gap <= 0:
        parser.error('Positive episode count, horizon and finite target gap are required')
    if any(not math.isfinite(v) for v in (args.kp, args.ki, args.kd)):
        parser.error('PID gains must be finite')
    if len(set(args.modes)) != len(args.modes):
        parser.error('--modes must not contain duplicates')
    if len({p.stem for p in args.checkpoints}) != len(args.checkpoints):
        parser.error('Checkpoint filenames must be distinct')
    policies = [(p, SafetyFilter.from_checkpoint(p)) for p in args.checkpoints]
    args.camera = args.nominal_cnn is not None or any(
        getattr(policy, 'observation_kind', 'state') == 'rgb' for _,policy in policies)
    if args.graphics is None:
        args.graphics = 'vulkan' if args.camera else 'null'
    if args.camera and args.graphics == 'null':
        parser.error('RGB policies require rendering: use --graphics vulkan or windowed')
    if 'safety_filter' in args.modes and any(not getattr(p,'has_safety_critic',True) for _,p in policies):
        parser.error('A supervised BC actor has no safety critic; select --modes actor_only or pid')
    dt = policies[0][1].settings['dt']
    contract = [policies[0][1].settings[k] for k in ('dt','safety_buffer_m','max_gap_m')]
    if any([p.settings[k] for k in ('dt','safety_buffer_m','max_gap_m')] != contract for _,p in policies):
        parser.error('Checkpoints must share timestep and failure bounds')
    if args.start_server and not args.server.is_file():
        parser.error('CARLA server launcher not found')
    # Refuse to overwrite previous results or silently mix experiments.
    args.output.mkdir(parents=True, exist_ok=False)
    import carla
    from run_gap_pid import GapPID, load_gap_estimator
    from collect_parallel import stop
    import torch
    torch.set_num_threads(4)
    estimate = load_gap_estimator(args.nominal_cnn) if args.nominal_cnn else None
    process, logfile, current, world = None, None, None, None
    summaries, reference_starts = [], {}
    client = carla.Client('127.0.0.1', args.port)
    client.set_timeout(30.)
    try:
        if args.start_server:
            for port in (args.port, args.port+1, args.port+2):
                with socket.socket() as sock:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    sock.bind(('0.0.0.0', port))
            logfile = (args.output/'server.log').open('w')
            env = dict(os.environ)
            env.setdefault('VK_ICD_FILENAMES', '/usr/share/vulkan/icd.d/nvidia_icd.json')
            graphics_flags = ['-RenderOffScreen', '-ResX=256', '-ResY=256']
            if args.graphics == 'null':
                graphics_flags.append('-nullrhi')
            process = subprocess.Popen([str(args.server.resolve()), *graphics_flags, '-quality-level=Epic',
                                        '-nosound', '-fps=20',
                                        f'-carla-rpc-port={args.port}', '-stdout', '-FullStdOutLogOutput'],
                                       stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True, env=env)
            client.set_timeout(2.)
            deadline = time.monotonic()+120
            while True:
                if process.poll() is not None:
                    raise RuntimeError('CARLA exited; inspect server.log')
                try:
                    # A pre-start connection can remain stale; use a fresh client
                    # on every attempt, matching the project's working launcher.
                    client = carla.Client('127.0.0.1', args.port)
                    client.set_timeout(5.)
                    version = client.get_server_version()
                    print(f'CARLA ready on {args.port}: {version}', flush=True)
                    break
                except RuntimeError:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(1.)
            client.set_timeout(120.)
        print('Loading evaluation map Town03', flush=True)
        world = client.load_world('Town03')
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = dt
        world.apply_settings(settings)
        scenarios = make_scenarios(world, args.episodes, args.seed, args.steps)
        write_json(args.output/'scenarios.json', scenarios)
        manifest = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items() if k != 'checkpoints'}
        manifest.update(checkpoints=[{'path':str(p), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p,_ in policies],
                        dt=dt, failure_bounds_m=contract[1:], gap_definition='original unsigned front/rear face-edge distance',
                        nominal=('GapPID with heading-to-leader steering; frozen CNN gap'
                                 if estimate else 'GapPID with heading-to-leader steering; oracle CARLA gap'),
                        leader=f'fixed saved route, {args.initial_speed or 2.:g} m/s throttle feedback, heading P control',
                        lane_departure='ego center outside any driving lane; count departure onsets',
                        sensor_drain='IMU frame barrier plus 0.2 s final collision-callback grace period',
                        camera_input=args.camera,
                        scope='development evaluation; frozen policy, no safety certification')
        if args.nominal_cnn:
            manifest['nominal_cnn_sha256'] = hashlib.sha256(args.nominal_cnn.read_bytes()).hexdigest()
        if args.leader_brake_test:
            manifest['leader_brake_event'] = dict(
                start_speed_mps=args.initial_speed or BRAKE_TEST_SPEED_MPS,
                speed_initialization='target velocity' if args.initial_speed else 'acceleration warmup',
                start_s=args.leader_brake_after, duration_s=args.leader_brake_duration, brake=1.)
        if args.stationary_leader_test:
            manifest['leader'] = 'stationary with full brake; ego accelerates toward it before handoff'
        write_json(args.output/'manifest.json', manifest)
        for checkpoint, policy in policies:
            destination = args.output/checkpoint.stem
            destination.mkdir()
            for scenario in scenarios:
                print(f'Running {checkpoint.stem} / {scenario["id"]} (sequential modes)', flush=True)
                mode_iterator = iter(args.modes)
                def reset(_id, seed, condition):
                    nonlocal current
                    if current is not None:
                        current.close()
                    mode = next(mode_iterator)
                    print(f'Viewing/running mode: {mode}', flush=True)
                    current = LiveScenario(client, args, dt, scenario)
                    pid = GapPID(args.target_gap, args.kp, args.ki, args.kd)
                    return current, NominalPID(pid, dt, estimate)
                results = run_matched_scenario(policy, reset, scenario_id=scenario['id'], seed=scenario['seed'],
                                              condition=args.condition, max_steps=args.steps,
                                              collect_measurements=lambda e,t: e.measurements(), modes=args.modes)
                for mode, result in results.items():
                    trace = result['trace']
                    start = trace['initial_pose'] + [trace['initial_gap_m']] + trace['initial_velocity_mps']
                    if scenario['id'] in reference_starts:
                        if not torch.allclose(torch.tensor(start), torch.tensor(reference_starts[scenario['id']]), atol=1e-3, rtol=0.):
                            raise ValueError('Initial physical state differs across checkpoints')
                    else:
                        reference_starts[scenario['id']] = start
                    result['summary']['checkpoint'] = checkpoint.stem
                    summaries.append(result['summary'])
                write_json(destination/f"{scenario['id']}.json", results)
                with (args.output/'summary.csv').open('w', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
                    writer.writeheader()
                    writer.writerows(summaries)
                print(f'{checkpoint.stem} {scenario["id"]}: '+', '.join(
                    f'{mode}: collision={r["summary"]["any_collision"]}, gap=[{r["summary"]["min_gap_m"]:.2f},{r["summary"]["max_gap_m"]:.2f}]'
                    for mode,r in results.items()), flush=True)
        aggregate = []
        for checkpoint, _ in policies:
            for mode in args.modes:
                rows = [r for r in summaries if r['checkpoint'] == checkpoint.stem and r['mode'] == mode]
                aggregate.append(dict(checkpoint=checkpoint.stem, mode=mode, episodes=len(rows),
                                      **{key+'_rate': sum(bool(r[key]) for r in rows)/len(rows)
                                         for key in ('any_collision', 'leader_collision', 'near_violation', 'far_violation')},
                                      lane_departure_episode_rate=sum(r['lane_departures'] > 0 for r in rows)/len(rows),
                                      mean_ego_travel_m=sum(r['ego_travel_m'] for r in rows)/len(rows),
                                      mean_intervention_rate=(sum(r['intervention_rate_eligible'] or 0 for r in rows)/len(rows)
                                                              if mode == 'safety_filter' else None)))
        write_json(args.output/'aggregate.json', aggregate)
        write_json(args.output/'complete.json', {'rollouts':len(summaries)})
    finally:
        try:
            if current is not None:
                current.close()
            if world is not None:
                # Do not mask the original failure by reconnecting to a crashed server.
                with suppress(RuntimeError):
                    client.set_timeout(3.)
                    live = client.get_world()
                    settings = live.get_settings()
                    settings.synchronous_mode = False
                    live.apply_settings(settings)
        finally:
            if process is not None:
                stop(process)
            if logfile is not None:
                logfile.close()


if __name__ == '__main__':
    main()
