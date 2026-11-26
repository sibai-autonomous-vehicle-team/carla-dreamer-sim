"""Raw-critic full-brake switching and paired evaluation accounting."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from training.evaluate_braking_critic import check_initial_state, filtered_action, run_rollout, summarize
from training.config import Settings


class BrakingEvaluationTests(unittest.TestCase):
    def test_nonpositive_score_applies_full_brake_and_retains_lane_steering(self):
        for score in (-.5, 0.):
            action, active = filtered_action((1.2, -.15), score, True)
            self.assertEqual(action, (-3., -.15))
            self.assertTrue(active)

    def test_positive_score_releases_braking(self):
        action, active = filtered_action((1.2, -.15), .01, True)
        self.assertEqual(action, (1.2, -.15))
        self.assertFalse(active)

    def test_fallback_request_maps_to_carla_full_brake_and_lane_steering(self):
        from car_dreamer.carla_base_env import CarlaBaseEnv
        env = SimpleNamespace(_config=SimpleNamespace(action=SimpleNamespace(discrete=False)))
        action, _ = filtered_action((1.2, -.15), -.5, True)
        control = CarlaBaseEnv.get_vehicle_control(env, action)
        self.assertEqual(control.throttle, 0.)
        self.assertEqual(control.brake, 1.)
        self.assertAlmostEqual(control.steer, .15)

    def test_pid_arm_uses_nominal_action_for_negative_scores(self):
        self.assertEqual(filtered_action((1.2, -.15), -.5, False), ((1.2, -.15), False))

    def test_nonfinite_score_is_an_evaluation_failure(self):
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            filtered_action((1., 0.), float('nan'), True)

    def test_pair_requires_matched_poses_and_velocities(self):
        initial = dict(pose=[1., 2.], velocity=[15., 0.])
        check_initial_state(initial, initial)
        with self.assertRaisesRegex(RuntimeError, 'velocity'):
            check_initial_state(dict(pose=[1., 2.], velocity=[15.1, 0.]), initial)

    def test_summary_retains_failures_and_paired_collision_counts(self):
        pid = dict(collision=True, lane_failure=False, ending='collision', intervention_steps=0,
                   minimum_gap_m=.1, initial=dict(gap_m=3.), first_intervention_s=None, trace=[])
        filtered = dict(collision=False, lane_failure=False, ending='braking_complete', intervention_steps=10,
                        minimum_gap_m=2., initial=dict(gap_m=3.), first_intervention_s=1., trace=[])
        pairs = [dict(episode=1, status='complete', pid=pid, filtered=filtered),
                 dict(episode=2, status='failed', error='IMU timeout')]
        result = summarize(pairs)
        self.assertEqual(result['requested_pairs'], 2)
        self.assertEqual(result['completed_pairs'], 1)
        self.assertEqual(result['pid']['collisions'], 1)
        self.assertEqual(result['filtered']['collisions'], 0)
        self.assertEqual(result['paired_collision_outcomes']['pid_collision_filtered_no_collision'], 1)
        self.assertEqual(result['paired_completion_outcomes']['pid_failure_filtered_complete'], 1)
        self.assertEqual(result['pid']['lower_buffer_violations'], 1)
        self.assertEqual(result['filtered']['lower_buffer_violations'], 0)
        self.assertEqual(result['failed_pairs'][0]['episode'], 2)

    def test_filtered_only_summary_has_twenty_single_rollouts(self):
        filtered = dict(collision=False, lane_failure=False, ending='braking_complete', intervention_steps=10,
                        minimum_gap_m=2., initial=dict(gap_m=3.), first_intervention_s=1., trace=[])
        pairs = [dict(episode=i, status='complete', filtered=filtered) for i in range(20)]
        result = summarize(pairs, modes=['filtered'])
        self.assertEqual(result['filtered']['episodes'], 20)
        self.assertEqual(result['filtered']['completed_braking'], 20)
        self.assertNotIn('pid', result)

    def test_realtime_hud_displays_decision_score_before_step_and_paces_whole_tick(self):
        frame = 0
        order = []
        env = SimpleNamespace(args=SimpleNamespace(external_pacing=True), events=[], leader_events=[],
                              ego=SimpleNamespace(id=1), validate_lanes=Mock(),
                              nominal_action=lambda pid: (1., .15),
                              get_rgb_observation=lambda frame: torch.zeros(224, 224, 3, dtype=torch.uint8),
                              measurements=lambda: dict(collision_events=[]),
                              display=SimpleNamespace(show=Mock()), current_chase='terminal image')
        def sample(env, dt):
            return frame, frame * dt, torch.zeros(12), 3., torch.ones(6)
        def step(action):
            nonlocal frame
            order.append(('step', action))
            frame += 1
        env.step = step
        env.render_decision = lambda *args: order.append(('render', args))
        model = lambda image, action: torch.tensor([-.25])
        with patch('training.evaluate_braking_critic.sample_state', sample), patch(
                'training.evaluate_braking_critic.time.monotonic', side_effect=[0., .04, .1, .15]), patch(
                'training.evaluate_braking_critic.time.sleep') as sleep:
            result = run_rollout(env, dict(target_gap=3., brake_after=.1), model, Settings(),
                                 torch.device('cpu'), (.9, .05, 1.), .1, True)
        self.assertEqual(order[0], ('render', (-.25, 'safety_filter', True)))
        self.assertEqual(order[1], ('step', (-3., .15)))
        self.assertEqual(result['transitions'], 2)
        self.assertAlmostEqual(sleep.call_args_list[1].args[0], .06)
        self.assertAlmostEqual(sleep.call_args_list[3].args[0], .05)
        self.assertEqual(env.display.show.call_args.kwargs['status_override'], 'Braking complete')


if __name__ == '__main__':
    unittest.main()
