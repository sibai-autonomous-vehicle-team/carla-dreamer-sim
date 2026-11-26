"""Collection planning, terminal alignment, and browser streaming checks."""
import json
import io
import math
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
from urllib.request import urlopen

import numpy as np
import torch

from training.collect_braking import BrowserMonitor, DatasetWriter, collect_episode, main, sample_plan
from training.braking_scenario import BrakingScenario, InvalidBrakingScenario, RouteTracker
from training.prepare_data import pack_episodes, read_conditions
from generate_dataset import save_atomic


class CollectionTests(unittest.TestCase):
    def run_mocked_collector(self, failures):
        payload, summary = self.episode()
        env = NS(close=Mock())
        world = NS(get_settings=lambda: NS(), apply_settings=Mock())
        client = NS(set_timeout=Mock(), load_world=lambda name: world, get_world=lambda: world)
        with TemporaryDirectory() as directory:
            root = Path(directory) / 'collection'
            args = NS(outdir=root, trajectories_per_condition=1, seed=42, speed_range=[12., 20.],
                      gap_range=[2., 5.], target_gap=3., brake_delay_range=[1.5, 2.5], brake_duration=2.,
                      conditions=['day_sunny'], kp=.9, ki=.05, kd=1., dt=.1, port=2700,
                      start_server=False, server=Path('unused-server'), headless=True,
                      real_time=False, collision_grace=0., resume=False)
            with patch('argparse.ArgumentParser.parse_args', return_value=args), patch.dict(
                    'sys.modules', {'carla': NS(Client=lambda *args: client)}), patch(
                    'training.collect_braking.make_scenarios', return_value=[dict(id='mock')]) as spawns, patch(
                    'training.collect_braking.BrakingScenario', side_effect=failures + [env]) as scenarios, patch(
                    'training.collect_braking.collect_episode', return_value=(payload, summary)), redirect_stdout(io.StringIO()):
                main()
            self.assertEqual(scenarios.call_count, len(failures) + 1)
            self.assertEqual(spawns.call_count, len(failures) + 1)
            self.assertEqual(json.loads((root / 'complete.json').read_text())['episodes'], 1)
            self.assertEqual(len(pack_episodes(read_conditions(root))['lengths']), 1)
            env.close.assert_called_once()

    def test_lane_rejections_replace_spawns_and_recover_beyond_thirty_attempts(self):
        self.run_mocked_collector([InvalidBrakingScenario('leader left the planned lane')] * 35)

    def test_persistent_simulator_errors_still_stop_after_thirty_failures(self):
        with self.assertRaisesRegex(RuntimeError, 'IMU timeout'):
            self.run_mocked_collector([RuntimeError('IMU timeout')] * 30)

    def test_collector_handoff_uses_requested_speed_range(self):
        env = BrakingScenario.__new__(BrakingScenario)
        env.args = NS(initial_speed=19.95, handoff_speed_bounds=[12., 20.])
        env.ego = NS(get_velocity=lambda: NS(length=lambda: 19.6))
        env.nonego = NS(get_velocity=lambda: NS(length=lambda: 19.7))
        env.validate_handoff_speed()
        env.nonego = NS(get_velocity=lambda: NS(length=lambda: 20.1))
        with self.assertRaisesRegex(InvalidBrakingScenario, 'outside requested range'):
            env.validate_handoff_speed()

    def tracker(self, route, x=0., y=0., yaw=0., speed=15.):
        position = NS(x=x, y=y, z=0.)
        transform = NS(location=position, rotation=NS(yaw=yaw))
        wheels = [NS(position=NS(x=x), max_steer_angle=70.) for x in (125., 125., -125., -125.)]
        vehicle = NS(get_transform=lambda: transform, get_location=lambda: position,
                     get_velocity=lambda: NS(length=lambda: speed),
                     get_physics_control=lambda: NS(wheels=wheels, steering_curve=[NS(x=0., y=1.), NS(x=10., y=.5)]))
        return RouteTracker(vehicle, route), transform

    def test_route_steering_tracks_lane_independently_of_leader(self):
        route = [[float(x), 0., 0.] for x in range(0, 201, 2)]
        straight, _ = self.tracker(route)
        right, _ = self.tracker(route, y=1.)
        left, _ = self.tracker(route, y=-1.)
        self.assertAlmostEqual(straight.steering(), 0.)
        self.assertLess(right.steering(), 0.)
        self.assertGreater(left.steering(), 0.)
        self.assertAlmostEqual(straight.steering_scale(5.), .75)
        self.assertAlmostEqual(straight.steering_scale(72.), .5)

    def test_route_tracker_negotiates_curves_in_bicycle_model(self):
        for speed in (12., 20.):
            for radius in (30., 80.):
                for direction in (-1, 1):
                    route = [[radius * math.sin(i / 100), direction * radius * (1 - math.cos(i / 100)), 0.]
                             for i in range(251)]
                    tracker, transform = self.tracker(route, speed=speed)
                    maximum_offset = 0.
                    for _ in range(100):
                        steer = tracker.steering()
                        angle = steer * tracker.max_angle * tracker.steering_scale(speed * 3.6)
                        heading = math.radians(transform.rotation.yaw)
                        transform.location.x += speed * math.cos(heading) * .02
                        transform.location.y += speed * math.sin(heading) * .02
                        transform.rotation.yaw += math.degrees(speed / tracker.wheelbase * math.tan(angle) * .02)
                        _, offset = tracker.project(transform.location)
                        maximum_offset = max(maximum_offset, offset)
                    self.assertLess(maximum_offset, .3)

    def test_both_cars_are_checked_against_the_planned_lane(self):
        env = BrakingScenario.__new__(BrakingScenario)
        env.carla = NS(LaneType=NS(Driving=1))
        env.map = NS(get_waypoint=lambda *args, **kwargs: NS(lane_width=3.5))
        env.ego = NS(id=1, get_location=lambda: None, bounding_box=NS(extent=NS(x=2., y=.9)),
                     get_transform=lambda: NS(rotation=NS(yaw=0.)))
        env.nonego = NS(id=2, get_location=lambda: None, bounding_box=NS(extent=NS(x=2., y=.9)),
                        get_transform=lambda: NS(rotation=NS(yaw=0.)))
        offsets = {1: 0., 2: 1.2}
        env.tracker = lambda actor: NS(project=lambda location: (10., offsets[actor.id]),
                                       point=lambda progress: ([10., 0., 0.], 0.))
        with self.assertRaisesRegex(RuntimeError, 'leader left the planned lane'):
            env.validate_lanes()
        offsets[2], offsets[1] = 0., 1.2
        with self.assertRaisesRegex(RuntimeError, 'ego left the planned lane'):
            env.validate_lanes()

    def test_speed_initialization_steers_both_vehicles(self):
        env = BrakingScenario.__new__(BrakingScenario)
        env.dt = .1
        env.carla = NS(VehicleControl=NS, Vector3D=NS)
        controls = []
        def vehicle(identity):
            actor = NS(id=identity, speed=0.)
            actor.get_transform = lambda: NS(get_forward_vector=lambda: NS(x=1., y=0., z=0.))
            actor.get_velocity = lambda: NS(length=lambda: actor.speed)
            actor.set_target_velocity = lambda velocity: setattr(actor, 'speed', velocity.x)
            actor.apply_control = lambda control: controls.append((identity, control.steer))
            return actor
        env.ego, env.nonego = vehicle(1), vehicle(2)
        env.tracker = lambda actor: NS(steering=lambda: actor.id * .1)
        env.validate_lanes = lambda: [0., 0.]
        env.world = NS(tick=lambda: None)
        env.initialize_speed(15.)
        self.assertEqual(controls, [(1, .1), (2, .2)])

    def test_leader_brakes_for_two_seconds_after_new_delay(self):
        env = BrakingScenario.__new__(BrakingScenario)
        env.carla = NS(VehicleControl=lambda **kwargs: NS(throttle=kwargs.get('throttle', 0.),
                                                        brake=kwargs.get('brake', 0.), steer=kwargs['steer']))
        env.dt, env.initial_frame = .1, 100
        env.args = NS(leader_brake_after=1.5, leader_brake_duration=2., initial_speed=15.)
        env.nonego = NS(get_velocity=lambda: NS(length=lambda: 14.))
        env.tracker = lambda actor: NS(steering=lambda: .2)
        for frame, expected_brake in ((114, 0.), (115, 1.), (134, 1.), (135, 0.)):
            env.world = NS(get_snapshot=lambda: NS(frame=frame))
            control = env.leader_control()
            self.assertEqual(control.brake, expected_brake)
            self.assertEqual(control.steer, .2)

    def test_plan_is_reproducible_and_uses_requested_ranges(self):
        args = NS(seed=42, conditions=['day_sunny', 'night_rainy'], trajectories_per_condition=10,
                  speed_range=[12., 20.], gap_range=[2., 8.5], target_gap=None,
                  brake_delay_range=[1.5, 2.5])
        scenarios = [dict(id=i) for i in range(20)]
        plan = sample_plan(args, scenarios)
        self.assertEqual(plan, sample_plan(args, scenarios))
        for entry in plan:
            self.assertTrue(12 <= entry['initial_speed'] <= 20)
            self.assertTrue(1.5 <= entry['brake_after'] <= 2.5)
            self.assertEqual(entry['target_gap'], entry['initial_gap'])
        self.assertEqual(sum(x['condition'] == 'day_sunny' for x in plan), 10)
        self.assertEqual(len(plan), 20)

    def episode(self, collision_tick=None, gap_range=None, speed_range=None):
        args = NS(dt=.1, kp=.9, ki=.05, kd=.15, n_traj=1, brake_duration=.2, collision_grace=0.)
        if gap_range is not None:
            args.gap_range = gap_range
        if speed_range is not None:
            args.speed_range = speed_range
        plan = dict(episode=0, initial_speed=15., target_gap=3., brake_after=.15, condition='day_sunny')
        env = NS(events=[], leader_events=[], tick=0, validate_lanes=lambda: [0., 0.],
                 nominal_action=lambda pid: (-3., 0.))
        env.ego = NS(id=1, get_control=lambda: NS(throttle=0., brake=1., steer=0.))
        env.nonego = NS(id=2, get_control=lambda: NS(throttle=0., brake=1., steer=0.))
        env.get_rgb_observation = lambda frame: torch.full((224, 224, 3), frame, dtype=torch.uint8)
        env.leader_control = lambda: NS(throttle=0., brake=1., steer=0.)
        def step(action):
            env.tick += 1
            if env.tick == collision_tick:
                env.events.append((env.tick, 2))
        env.step = step
        env.measurements = lambda: dict(collision_events=env.events, lane_departure_frames=[])
        def sample(_env, dt):
            return env.tick, env.tick * dt, torch.zeros(12), 3., torch.ones(6)
        with patch('training.collect_braking.sample_state', sample):
            return collect_episode(env, plan, args, None)

    def test_normal_end_contains_one_more_observation_than_actions(self):
        episode, summary = self.episode()
        self.assertEqual(summary['ending'], 'braking_complete')
        self.assertEqual(len(episode['actions']), 4)
        self.assertEqual(episode['frames'].tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(episode['images'].shape[0], 5)
        self.assertFalse(episode['collision'].any())

    def test_collision_ends_before_another_action(self):
        episode, summary = self.episode(collision_tick=2)
        self.assertEqual(summary['ending'], 'collision')
        self.assertEqual(episode['frames'].tolist(), [0, 1, 2])
        self.assertEqual(episode['collision'].tolist(), [False, True])
        self.assertEqual(len(episode['actions']), 2)

    def test_writer_matches_original_layout_types_and_row_alignment(self):
        payload, summary = self.episode(collision_tick=2)
        payload['summary'] = summary
        payload['distances'] = torch.tensor([5., 4., 3.])
        payload['states'] = torch.arange(36, dtype=torch.float32).reshape(3, 12)
        with TemporaryDirectory() as directory:
            writer = DatasetWriter(directory, ['day_sunny', 'night_rainy'])
            writer.write(payload, 'day_sunny', 0)
            writer.write(payload, 'night_rainy', 0)
            root = Path(directory)
            groups = read_conditions(root)
            packed = pack_episodes(groups)
            self.assertEqual(packed['lengths'].tolist(), [2, 2])
            for condition in groups:
                fields = groups[condition]
                self.assertEqual(fields['distances'][0].tolist(), [4., 3.])
                self.assertEqual(fields['costs'][0].tolist(), [0., 1.])
                self.assertTrue(torch.equal(fields['states'][0], payload['states'][:-1]))
                self.assertTrue(torch.equal(fields['actions'][0], payload['actions']))
                image = torch.load(root / condition / 'obses/episode_000.pth', weights_only=True)
                self.assertEqual(image.shape, (2, 224, 224, 3))
                self.assertEqual(image.dtype, torch.uint8)
                self.assertTrue(torch.equal(image, payload['images'][:-1]))
                metadata = torch.load(root / condition / 'metadata/episode_000.pth', weights_only=True)
                self.assertTrue(torch.equal(metadata['terminal_image'], payload['images'][-1]))
                self.assertTrue(torch.equal(metadata['terminal_state'], payload['states'][-1]))
                self.assertEqual(metadata['initial_gap'].item(), 5.)
            resumed = DatasetWriter(directory, ['day_sunny', 'night_rainy'])
            self.assertEqual(resumed.completed('day_sunny'), 1)
            self.assertEqual(resumed.summary('night_rainy', 0), summary)
            self.assertFalse(list(root.glob('episode_*.pt')))

    def test_resume_recovers_interrupted_numerical_write(self):
        payload, summary = self.episode()
        payload['summary'] = summary
        with TemporaryDirectory() as directory:
            writer = DatasetWriter(directory, ['day_sunny'])
            writer.write(payload, 'day_sunny', 0)
            def interrupt_commit(data, path):
                if Path(path).name == 'seq_lengths.pth':
                    raise OSError('interrupted before commit')
                save_atomic(data, path)
            with patch('training.collect_braking.save_atomic', side_effect=interrupt_commit):
                with self.assertRaisesRegex(OSError, 'interrupted'):
                    writer.write(payload, 'day_sunny', 1)
            resumed = DatasetWriter(directory, ['day_sunny'])
            self.assertEqual(resumed.completed('day_sunny'), 1)
            self.assertEqual(len(read_conditions(Path(directory))['day_sunny']['actions']), 1)
            resumed.write(payload, 'day_sunny', 1)
            self.assertEqual(pack_episodes(read_conditions(Path(directory)))['lengths'].tolist(), [4, 4])

    def test_out_of_range_handoffs_receive_no_training_label(self):
        with self.assertRaisesRegex(InvalidBrakingScenario, 'starting bumper gap'):
            self.episode(gap_range=[4., 5.])
        with self.assertRaisesRegex(InvalidBrakingScenario, 'starting speed'):
            self.episode(speed_range=[12., 20.])

    def test_browser_reuses_template_and_serves_latest_frame(self):
        monitor = BrowserMonitor(0)
        try:
            base = f'http://127.0.0.1:{monitor.server.server_port}'
            with urlopen(base, timeout=5) as response:
                self.assertIn(b'CarDreamer Visualization', response.read())
            monitor.render(np.zeros((224, 224, 3), dtype=np.uint8), dict(gap_m=3., phase='following'))
            with urlopen(base + '/stream', timeout=5) as response:
                data = json.loads(response.readline().decode().removeprefix('data: '))
                self.assertEqual(data['info']['gap_m'], 3.)
                self.assertEqual(data['images'][0]['key'], 'visual')
        finally:
            monitor.close()


if __name__ == '__main__':
    unittest.main()
