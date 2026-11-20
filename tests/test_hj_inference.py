"""Deployment contract checks that run without a CARLA server or archived datasets."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from training.closed_loop import read_snapshot, run_episode, run_matched_scenario, summarize_episode
from training.config import LOCAL_DIR
from training.dataset import HJ_FEATURES
from training.models import Actor, Critic, SafetyFilter


def checkpoint():
    torch.manual_seed(12)
    actor = Actor(8, (6,), [-3, -1], [3, 1])
    critic = Critic(8, 2, (6,))
    return {
        "format_version": 2, "objective": "discounted_two_sided_first_hit",
        "action_convention": "effective_requested_longitudinal_and_requested_steer",
        "features": HJ_FEATURES, "state_mean": torch.arange(8).float(),
        "state_std": torch.arange(1, 9).float(),
        "actor": actor.state_dict(), "critic": critic.state_dict(),
        "settings": {"switching_threshold": 0., "longitudinal_saturation": 3.,
                     "hidden_sizes": [6], "action_low": [-3, -1], "action_high": [3, 1],
                     "safety_buffer_m": 1., "max_gap_m": 30., "dt": .1},
    }


class InferenceTests(unittest.TestCase):
    def test_raw_features_reload_from_fresh_interpreter(self):
        saved = checkpoint()
        policy = SafetyFilter(saved)
        state = [4., 7., 1., 0., 1., 3., -1., .3]
        reference = (torch.tensor(state) - saved["state_mean"]) / saved["state_std"]
        self.assertTrue(torch.equal(policy.normalize(state), reference))
        expected = [list(policy.predict_safe_action(state)), policy.score_action(state, [-2., .5])]
        LOCAL_DIR.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=LOCAL_DIR) as temp:
            path = Path(temp) / "deployment.pt"
            torch.save(saved, path)
            code = ("import json,sys; from training.models import SafetyFilter; "
                    "p=SafetyFilter.from_checkpoint(sys.argv[1]); s=json.loads(sys.argv[2]); "
                    "print(json.dumps([p.predict_safe_action(s),p.score_action(s,[-2.,.5])]))")
            result = subprocess.check_output([sys.executable, "-B", "-c", code, str(path), json.dumps(state)],
                                             cwd=LOCAL_DIR.parents[1], text=True)
        self.assertEqual(expected, json.loads(result))
        self.assertTrue(all(not p.requires_grad for m in (policy.actor, policy.critic) for p in m.parameters()))

    def test_equality_intervenes_raw_nominal_retained_above_threshold(self):
        saved = checkpoint()
        for w in saved["critic"].values():
            w.zero_()
        state, nominal = [1.] * 8, [4., .2]
        policy = SafetyFilter(saved)
        decision = policy.filter_action(state, nominal)
        self.assertTrue(decision.intervened)
        self.assertEqual(decision.action, policy.predict_safe_action(state))
        self.assertEqual(decision.nominal_q, 0.)
        saved["critic"]["network.2.bias"].fill_(1.)
        retained = SafetyFilter(saved).filter_action(state, nominal)
        self.assertFalse(retained.intervened)
        self.assertEqual(retained.action, tuple(nominal))
        other = SafetyFilter(checkpoint())
        self.assertEqual(other.score_action(state, [4., .2]), other.score_action(state, [3., .2]))
        with self.assertRaises(ValueError):
            other.score_action(state, [0., 1.5])
        with self.assertRaises(ValueError):
            SafetyFilter({**saved, "objective": "old"})

    def test_actual_wrapper_control_mapping(self):
        # Load the real method with only its import-time simulator types replaced.
        path = LOCAL_DIR.parents[1] / "car_dreamer/carla_base_env.py"
        spec = importlib.util.spec_from_file_location("_hj_test.carla_base_env", path)
        module = importlib.util.module_from_spec(spec)
        modules = {"carla": SimpleNamespace(VehicleControl=SimpleNamespace, Actor=object),
                   "gym": SimpleNamespace(Env=object, spaces=object),
                   "_hj_test.toolkit": SimpleNamespace(EnvMonitorOpenCV=object, Observer=object, WorldManager=object)}
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(module)
        env = SimpleNamespace(_config=SimpleNamespace(action=SimpleNamespace(discrete=False)))
        for request, expected in [([-1.5,.25], (0.,.5,-.25)), ([4.,-.5], (1.,0.,.5)),
                                  ([-4.,0.], (0.,1.,0.)), ([0.,0.], (0.,0.,0.))]:
            control = module.CarlaBaseEnv.get_vehicle_control(env, request)
            self.assertEqual((control.throttle, control.brake, control.steer), expected)

    def test_same_frame_pose_and_clearance(self):
        tf = lambda x, p: SimpleNamespace(location=SimpleNamespace(x=x, y=0., z=0.),
                                          rotation=SimpleNamespace(pitch=p, yaw=0., roll=0.))
        transforms = {4: tf(1., .5), 9: tf(8., 2.)}
        snap = SimpleNamespace(frame=30, timestamp=SimpleNamespace(delta_seconds=.1, elapsed_seconds=3.),
                               find=lambda i: SimpleNamespace(get_transform=lambda: transforms[i]))
        world = SimpleNamespace(get_snapshot=lambda: snap)
        env = SimpleNamespace(ego=SimpleNamespace(id=4, get_world=lambda: world), nonego=SimpleNamespace(id=9))
        def measured_gap(a,b):
            self.assertIs(a, transforms[4])
            self.assertIs(b, transforms[9])
            return 2.75
        env.get_distance = measured_gap
        frame, _, pose, gap = read_snapshot(env, .1)
        self.assertEqual((frame, pose[0], pose[6], pose[9], gap), (30, 1., 8., 2., 2.75))

    def test_live_modes_warmup_current_gap_and_realized_outcome_scoring(self):
        policy = SafetyFilter(checkpoint())
        for mode in ("pid", "actor_only", "safety_filter"):
            nominal = (2., .2)
            seen, sent = [], []
            world = SimpleNamespace(get_snapshot=lambda: SimpleNamespace(frame=len(sent)),
                                    get_settings=lambda: SimpleNamespace(synchronous_mode=True, fixed_delta_seconds=.1))
            env = SimpleNamespace(ego=SimpleNamespace(get_world=lambda: world), nonego=SimpleNamespace(id=9),
                                  _config=SimpleNamespace(action=SimpleNamespace(discrete=False)))
            def step(action):
                sent.append(action)
                return None, 0., False, {}
            env.step = step
            def snapshot(_env, _dt):
                i = len(sent)
                pose = torch.zeros(12)
                pose[0], pose[6] = i, 10.
                return i, i*.1, pose, [2., .5, 31.][i]
            def actor(state):
                seen.append(state)
                return (-1., -.2)
            with patch("training.closed_loop.read_snapshot", snapshot), patch.object(policy, "predict_safe_action", actor), patch.object(policy, "score_action", return_value=-1.):
                trace = run_episode(env, policy, lambda _: nominal, mode=mode, scenario_id="fixture", seed=7,
                                    condition="day", max_steps=2)
            self.assertEqual(sent[0], nominal)
            self.assertEqual(sent[1], nominal if mode=="pid" else (-1., -.2))
            if seen:
                self.assertEqual(seen[0][0], .5)
                self.assertEqual(seen[0][5], 10.)
            result = summarize_episode(trace, collision_events=[(2,9),(6,8)], lane_departure_frames=[])
            self.assertTrue(result["near_violation"] and result["far_violation"] and result["leader_collision"])
            self.assertEqual(result["contact_event_messages"], 1)
            self.assertEqual(result["intervention_rate_eligible"], 1. if mode=="safety_filter" else None)
            self.assertIsNone(summarize_episode(trace)["any_collision"])

    def test_matched_evaluation_detects_reset_drift(self):
        policy = SafetyFilter(checkpoint())
        resets = []
        def reset(scenario, seed, weather):
            resets.append((scenario, seed, weather))
            pose = torch.zeros(12)
            pose[6] = 9.
            pose[0] = 0. if len(resets) < 3 else 2.
            vel = SimpleNamespace(x=0., y=0., z=0.)
            snap = SimpleNamespace(frame=0, find=lambda _: SimpleNamespace(get_velocity=lambda: vel))
            world = SimpleNamespace(get_snapshot=lambda: snap,
                                    get_settings=lambda: SimpleNamespace(synchronous_mode=True, fixed_delta_seconds=.1))
            ego = SimpleNamespace(id=1, type_id="ego", get_world=lambda: world)
            env = SimpleNamespace(ego=ego, nonego=SimpleNamespace(id=2, type_id="leader"), pose=pose,
                                  _config=SimpleNamespace(action=SimpleNamespace(discrete=False)))
            def step(_):
                snap.frame += 1
                return None, 0., False, {}
            env.step = step
            return env, lambda _: (0.,0.)
        def snapshot(env, dt):
            frame = env.ego.get_world().get_snapshot().frame
            return frame, frame*dt, env.pose, 5.
        with patch("training.closed_loop.read_snapshot", snapshot), self.assertRaisesRegex(ValueError, "reset"):
            run_matched_scenario(policy, reset, scenario_id="fixture", seed=22, condition="rain", max_steps=1)
        self.assertEqual(resets, [("fixture",22,"rain")]*3)
