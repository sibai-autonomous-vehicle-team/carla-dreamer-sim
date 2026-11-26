"""Direct safety actor execution without a nominal controller dependency."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from torch import nn

from training.train_online_hj import SafetyPolicy, behavior_action, parse_args
from training.evaluate_online_hj import evaluate


class SafetyOnlyTests(unittest.TestCase):
    def test_cli_does_not_require_pid_files(self):
        with patch('sys.argv', ['hj', '--outdir', '/tmp/unused-safety-test',
                  '--performance-policy', 'safety', '--pid-root', '/missing-pids']):
            args = parse_args()
        self.assertEqual(args.performance_policy, 'safety')

    def test_warmup_then_actor_without_gate_or_extra_random_episodes(self):
        args = NS(performance_policy='safety', random_steps=1000)
        learner = Mock()
        explorer = NS(action=Mock(return_value=2.))
        warmup = behavior_action(learner, None, -2., explorer, 999, args, False)
        direct = behavior_action(learner, None, -2., explorer, 1000, args, True)
        self.assertEqual(warmup['action'], 2.)
        self.assertEqual(direct['action'], -2.)
        self.assertEqual(direct['source'], 'safety_actor')
        self.assertFalse(direct['gate_active'])
        self.assertFalse(direct['intervened'])
        learner.score.assert_not_called()
        self.assertEqual(explorer.action.call_count, 1)

    def test_adapter_selects_deterministic_evaluation(self):
        learner = NS(action=Mock(return_value=-2.))
        policy = SafetyPolicy(learner, 6.)
        self.assertEqual(policy.propose('observation', .1, deterministic=True), (-2., None))
        learner.action.assert_called_once_with('observation', deterministic=True)

    def test_evaluation_runs_only_safety_and_tracks_repeated_checkpoints(self):
        learner = NS(actor=nn.Linear(1,1), critics=nn.Linear(1,1),
                     targets=nn.Linear(1,1), updates=100)
        baseline = dict(collision=False, horizon_complete=True)
        case = dict(case=0, start=dict(challenge=False), braking_baseline=baseline)
        record = dict(valid=True, collision=False, failure=False, lane_departure=False,
            horizon_complete=True, initial_gap=3., initial_ego_speed=15.,
            initial_leader_speed=15., intervention_fraction=0., mean_speed=10.,
            mean_absolute_gap_error=2.)
        with TemporaryDirectory() as directory:
            args = NS(outdir=Path(directory), performance_policy='safety', switching_threshold=1.)
            (args.outdir/'evaluation_suite.json').write_text(json.dumps([case]))
            with patch('training.evaluate_online_hj.rollout', return_value=record.copy()) as rollout, \
                 patch('training.plot_online_hj_evaluation.plot_evaluations'), \
                 redirect_stdout(io.StringIO()):
                first = evaluate(learner, {'safety':object()}, None, None, None, args, 5000)
                second = evaluate(learner, {'safety':object()}, None, None, None, args, 10000)
            self.assertEqual([c.args[6] for c in rollout.call_args_list], ['safety','safety'])
            self.assertEqual(first['paired_cases'], 1)
            self.assertEqual(first['modes']['safety']['episodes'], 1)
            self.assertEqual(first['modes']['pid']['episodes'], 0)
            self.assertEqual(second['change_from_previous']['collision_rate_delta'], {'safety':0.})


if __name__ == '__main__': unittest.main()
