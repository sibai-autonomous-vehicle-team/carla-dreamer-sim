"""Harness scenario and geometry contract checks without a CARLA server."""
import unittest
import queue
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from run_gap_pid import GapPID
from training.evaluate_closed_loop import LiveScenario, NominalPID, make_scenarios


class Waypoint:
    def __init__(self, x):
        self.road_id, self.section_id, self.lane_id, self.s = 1, 0, 1, x
        self.is_junction = False
        self.transform = NS(location=NS(x=x, y=0., z=0.), rotation=NS(pitch=0., yaw=0., roll=0.))

    def next(self, distance):
        return [Waypoint(self.s + distance)]


class RunnerTests(unittest.TestCase):
    def test_imu_timeout_uses_the_collectors_retryable_error_type(self):
        live = LiveScenario.__new__(LiveScenario)
        live.args = NS(graphics='vulkan', camera=False)
        live.world = NS(tick=lambda: 42)
        live.imu_frames = Mock()
        live.imu_frames.get.side_effect = queue.Empty
        with self.assertRaisesRegex(RuntimeError, 'IMU timeout waiting for frame 42') as caught:
            live.tick()
        self.assertIsInstance(caught.exception.__cause__, queue.Empty)

    def test_imu_wait_drains_older_frames_and_accepts_current_frame(self):
        live = LiveScenario.__new__(LiveScenario)
        live.args = NS(graphics='vulkan', camera=False)
        live.display = None
        live.world = NS(tick=lambda: 42)
        live.imu_frames = Mock()
        live.imu_frames.get.side_effect = [40, 41, 42]
        self.assertEqual(live.tick(), 42)

    def test_moving_start_compensates_for_drag_before_handoff(self):
        live = LiveScenario.__new__(LiveScenario)
        class Vehicle:
            def __init__(self):
                self.command, self.speed = 0., 0.
            def apply_control(self, control):
                pass
            def get_transform(self):
                return NS(get_forward_vector=lambda: NS(x=1., y=0., z=0.))
            def set_target_velocity(self, velocity):
                self.command = velocity.x
            def get_velocity(self):
                return NS(length=lambda: self.speed)
        live.ego, live.nonego = Vehicle(), Vehicle()
        def tick():
            for actor in (live.ego, live.nonego):
                actor.speed = actor.command - .2
        live.world = NS(tick=tick)
        live.carla = NS(VehicleControl=lambda: NS(), Vector3D=lambda **values: NS(**values))
        live.dt = .1
        live.initialize_speed(15.)
        self.assertAlmostEqual(live.ego.speed, 15., delta=.05)
        self.assertAlmostEqual(live.nonego.speed, 15., delta=.05)

    def test_high_speed_leader_brakes_at_configured_times(self):
        live = LiveScenario.__new__(LiveScenario)
        snapshot = NS(frame=4)
        live.world = NS(get_snapshot=lambda: snapshot)
        live.carla = NS(VehicleControl=lambda **values: NS(**values))
        live.args = NS(stationary_leader_test=False, leader_brake_test=True,
                       initial_speed=15., leader_brake_after=.5, leader_brake_duration=2.)
        live.nonego = NS(get_transform=lambda: NS(location=NS(x=0., y=0.), rotation=NS(yaw=0.)),
                         get_velocity=lambda: NS(x=14.9, y=0.))
        live.route, live.route_index = [[0., 0.], [20., 0.]], 0
        live.initial_frame, live.dt = 0, .1
        self.assertEqual(live.leader_control().throttle, 2 / 3)
        for frame in (5, 24):
            snapshot.frame = frame
            self.assertEqual(live.leader_control().brake, 1.)
        snapshot.frame = 25
        self.assertEqual(live.leader_control().throttle, 2 / 3)

    def test_nominal_cnn_drives_pid_from_synchronized_image(self):
        image = torch.zeros(224, 224, 3, dtype=torch.uint8)
        frames = []
        def observation(frame):
            frames.append(frame)
            return image
        def oracle_gap():
            self.fail('CNN nominal controller requested the oracle gap')
        env = NS(ego=NS(get_world=lambda: NS(get_snapshot=lambda: NS(frame=37))),
                 get_rgb_observation=observation, get_distance=oracle_gap)
        controller = NominalPID(GapPID(5., .5, 0., .5), .1, lambda _: 7.)
        with patch('run_gap_pid.steering_command', return_value=.2):
            self.assertEqual(controller(env), (1., .2))
        self.assertEqual(frames, [37])
        self.assertEqual(controller.gap_m, 7.)
        self.assertEqual(controller.image_frame, 37)

    def test_nominal_oracle_path_and_nonfinite_cnn(self):
        env = NS(get_distance=lambda: 7.)
        controller = NominalPID(GapPID(5., .5, 0., .5), .1)
        with patch('run_gap_pid.steering_command', return_value=0.):
            self.assertEqual(controller(env), (1., 0.))
        self.assertIsNone(controller.image_frame)
        env = NS(ego=NS(get_world=lambda: NS(get_snapshot=lambda: NS(frame=1))),
                 get_rgb_observation=lambda _: torch.zeros(224, 224, 3, dtype=torch.uint8))
        controller = NominalPID(GapPID(5., .5, 0., .5), .1, lambda _: float('nan'))
        with self.assertRaisesRegex(ValueError, 'Nonfinite nominal gap'):
            controller(env)

    def test_saved_scenarios_repeat_and_preserve_ten_metre_reset(self):
        world = NS(get_map=lambda: NS(generate_waypoints=lambda _: [Waypoint(0.), Waypoint(100.)]))
        a = make_scenarios(world, 2, 23, 20)
        self.assertEqual(a, make_scenarios(world, 2, 23, 20))
        self.assertEqual(len(a), 2)
        for scenario in a:
            self.assertEqual(scenario['leader'][0]-scenario['ego'][0], 10.)
            self.assertEqual(scenario['ego'][2], .5)
            self.assertGreater(len(scenario['route']), 30)

    def test_sensor_cleanup_continues_if_an_actor_is_already_gone(self):
        stopped = []
        class Actor:
            def stop(self):
                stopped.append('stop')
            def destroy(self):
                stopped.append('destroy')
                raise RuntimeError('already gone')
        live = LiveScenario.__new__(LiveScenario)
        live.actors = [Actor(), Actor()]
        live.close()
        self.assertEqual(stopped, ['stop', 'destroy', 'stop', 'destroy'])
        self.assertEqual(live.actors, [])


if __name__ == '__main__':
    unittest.main()
