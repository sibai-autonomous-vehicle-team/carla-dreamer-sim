"""Pairing prevents invalid or mismatched resets biasing controller scores."""
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from training.compare_generalist_specialist import ARMS, ContextSelectorPID, summarize


class ComparisonTests(unittest.TestCase):
    def rows(self, case=0):
        return [dict(condition='day_cloudy',case=case,arm=arm,challenge=False,valid=True,
                     initial_gap=5.,initial_ego_speed=12.,initial_leader_speed=12.,
                     collision=False,lane_departure=False,horizon_complete=True,
                     mean_absolute_gap_error=1.,mean_speed=10.,intervention_fraction=0.)
                for arm in ARMS]

    def test_uses_same_cases_for_all_arms(self):
        good = self.rows()
        good[1].update(collision=True,horizon_complete=False,intervention_fraction=.2)
        invalid = self.rows(1)
        invalid[2] = dict(condition='day_cloudy',case=1,arm=ARMS[2],challenge=False,valid=False)
        result = summarize(good+invalid)
        self.assertEqual(result['paired_cases'],1)
        self.assertEqual(len(result['excluded']),1)
        self.assertEqual(result['arms']['generalist_filtered']['collisions'],1)
        self.assertTrue(all(r['episodes']==1 for r in result['arms'].values()))

    def test_missing_arms_and_reset_drift_exclude_whole_case(self):
        drift = self.rows()
        drift[0]['initial_gap'] += .31
        result = summarize(drift+self.rows(1)[:2])
        self.assertEqual(result['paired_cases'],0)
        self.assertEqual(len(result['excluded']),2)
        self.assertTrue(all(r['collision'] is None for r in result['arms'].values()))

    def test_selector_keeps_integral_and_resets_derivative_every_two_seconds(self):
        choices = iter([{'day_sunny':.9,'day_cloudy':.1},{'day_sunny':.1,'day_cloudy':.9}])
        selector = NS(score=lambda condition,gap:next(choices))
        policy = ContextSelectorPID(selector,NS(gap=6.),
                                    {'day_sunny':NS(gap=8.),'day_cloudy':NS(gap=7.)},'day_cloudy')
        with patch.object(ContextSelectorPID,'estimate',side_effect=lambda p,o:p.gap):
            for _ in range(20): policy.propose(None,.1)
            action,gap = policy.propose(None,.1)
        self.assertAlmostEqual(action,1.105)
        self.assertEqual(gap,7.)
        self.assertEqual(policy.switches,1)
        self.assertEqual(dict(policy.selections),{'day_sunny':20,'day_cloudy':1})


if __name__ == '__main__':
    unittest.main()
