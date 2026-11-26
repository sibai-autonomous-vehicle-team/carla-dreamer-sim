"""Frozen evaluation, paired comparisons, event labels and trainer isolation."""
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from training.evaluate_online_hj import isolated_evaluation, summarize, evaluate


class EvaluationTests(unittest.TestCase):
    def test_restores_rng_and_nested_modes_even_on_error(self):
        learner=NS(actor=nn.Sequential(nn.Linear(1,1),nn.Dropout()),critics=nn.Linear(1,1),targets=nn.Linear(1,1))
        learner.goal_actor=nn.Linear(1,1); learner.goal_critics=nn.Linear(1,1); learner.goal_targets=nn.Linear(1,1)
        learner.actor.train(); learner.actor[0].eval(); learner.targets.eval()
        random.seed(8); np.random.seed(8); torch.manual_seed(8)
        expected=(random.random(),np.random.rand(),torch.rand(1))
        random.seed(8); np.random.seed(8); torch.manual_seed(8)
        with self.assertRaisesRegex(RuntimeError,'deliberate'):
            with isolated_evaluation(learner):
                self.assertFalse(torch.is_grad_enabled())
                self.assertFalse(learner.actor.training)
                self.assertFalse(learner.goal_actor.training)
                random.random(); np.random.rand(); torch.rand(1)
                raise RuntimeError('deliberate')
        self.assertTrue(learner.actor.training); self.assertFalse(learner.actor[0].training)
        self.assertFalse(learner.targets.training)
        self.assertTrue(learner.goal_actor.training)
        self.assertEqual(random.random(),expected[0]); self.assertEqual(np.random.rand(),expected[1])
        self.assertTrue(torch.equal(torch.rand(1),expected[2]))

    def test_automatic_evaluation_runs_four_modes_without_training_and_logs_false_safe(self):
        world=NS(frame=100)
        world.get_snapshot=lambda:NS(frame=world.frame,timestamp=NS(delta_seconds=.1,elapsed_seconds=world.frame*.1),
            find=lambda actor:NS(get_velocity=lambda:NS(x=15.,y=0.,z=0.)))
        instances=[]
        class Env:
            def __init__(self,*args,**kwargs):
                self.world=world; self.ego=NS(id=1); self.nonego=NS(id=2)
                self.events=[]; self.departures=[]; self.leader_events=[]; self.closed=False
                instances.append(self)
            def tracker(self,actor):return NS(steering=lambda:0.)
            def get_rgb_observation(self,frame):return torch.ones(224,224,3,dtype=torch.uint8)
            def step(self,action):
                world.frame+=1
                if action[0]>0: self.events.append((world.frame,2))
            def measurements(self):return dict(collision_events=self.events,lane_departure_frames=[])
            def close(self):self.closed=True
        learner=NS(actor=nn.Linear(1,1),critics=nn.Linear(1,1),targets=nn.Linear(1,1),updates=7,
            score=lambda obs,action:.5,action=lambda obs,deterministic:-3.,
            filter_action=lambda obs,action:dict(action=-3.,intervened=True))
        weights=learner.actor.weight.detach().clone()
        nominal=NS(reset=lambda:None,profile=dict(target_gap=6.),propose=lambda obs,dt:(1.,3.))
        case=dict(case=0,seed=100,start=dict(gap=3.,closing_speed=0.,leader_speed=15.,ego_speed=15.,challenge=False),
                  brake_after=1.,scenario={})
        with TemporaryDirectory() as directory, ExitStack() as stack:
            args=NS(outdir=Path(directory),condition='day_cloudy',brake_duration=2.,speed_range=[10.,20.],
                history_frames=4,dt=.1,episode_steps=5,collision_grace=0.,safety_objective='collision',
                max_gap=9.,max_separation=50.,event_failure_margin=1.,switching_threshold=.25)
            stack.enter_context(patch('training.evaluate_online_hj.make_suite',return_value=[case]))
            stack.enter_context(patch('training.closed_loop.read_snapshot',side_effect=lambda *args:(world.frame,0.,None,3.)))
            stack.enter_context(patch('training.train_online_hj.planned_lane_status',return_value=dict(inside=True)))
            with redirect_stdout(io.StringIO()):
                summary=evaluate(learner,{'day_cloudy':nominal},None,world,Env,args,5000)
            self.assertEqual(summary['paired_cases'],1)
            self.assertEqual(summary['modes']['pid']['collision'],1.)
            self.assertEqual(summary['modes']['safety']['collision'],0.)
            self.assertEqual(summary['modes']['filter']['intervention_fraction'],1.)
            probe=summary['modes']['pid_then_safety']
            self.assertEqual(probe['false_safe_collisions'],1)
            self.assertEqual(probe['missed_collision_fraction'],1.)
            self.assertEqual(summary['modes']['safety']['horizon_complete'],1.)
            self.assertTrue(all(env.closed for env in instances)); self.assertEqual(len(instances),4)
            self.assertTrue(torch.equal(weights,learner.actor.weight)); self.assertEqual(learner.updates,7)
            self.assertEqual(len((args.outdir/'evaluation_rollouts.jsonl').read_text().splitlines()),4)
            self.assertEqual(json.loads((args.outdir/'evaluation_latest.json').read_text())['step'],5000)
            self.assertTrue((args.outdir/'evaluation_plot.png').read_bytes().startswith(b'\x89PNG'))
            dashboard=(args.outdir/'evaluation_dashboard.html').read_text()
            self.assertIn('http-equiv="refresh"',dashboard)
            self.assertIn('PID profile: day_cloudy',dashboard)
            self.assertIn('Ordinary starts',dashboard)
            self.assertEqual(list(args.outdir.glob('*.pt')),[])
            rows=[json.loads(line) for line in (args.outdir/'evaluation_rollouts.jsonl').read_text().splitlines()]
            rows[0]['valid']=False
            self.assertEqual(summarize(rows)['paired_cases'],0)
            self.assertIsNone(summarize(rows)['modes']['safety']['collision'])
            rows[0]['valid']=True; rows[0]['initial_gap']=4.
            self.assertEqual(summarize(rows)['paired_cases'],0)
            with redirect_stdout(io.StringIO()):
                second=evaluate(learner,{'day_cloudy':nominal},None,world,Env,args,10000)
            self.assertTrue(second['change_from_previous']['same_matched_cases'])
            self.assertEqual(second['change_from_previous']['collision_rate_delta'],
                             dict(pid=0.,safety=0.,filter=0.,pid_then_safety=0.))
            self.assertIn('10,000',(args.outdir/'evaluation_dashboard.html').read_text())
            from training.online_sage import GoalPolicy
            flags=[]
            learner.goal_action=lambda obs,deterministic=False: (flags.append(deterministic) or 1.)
            learner.goal_actor=nn.Linear(1,1); learner.goal_critics=nn.Linear(1,1); learner.goal_targets=nn.Linear(1,1)
            args.outdir=root=Path(directory)/'goal_run'; root.mkdir()
            with redirect_stdout(io.StringIO()):
                goal_summary=evaluate(learner,{'goal':GoalPolicy(learner,6.)},None,world,Env,args,100)
            self.assertEqual(goal_summary['nominal_kind'],'goal')
            self.assertTrue(flags); self.assertTrue(all(flags))
            dashboard=(root/'evaluation_dashboard.html').read_text()
            self.assertIn('SAGE (goal + safety)',dashboard)
            self.assertIn('goal-policy action',dashboard)

    def test_plot_missing_rates_stay_missing_and_all_checkpoints_are_shown(self):
        from training.plot_online_hj_evaluation import evaluation_figure
        from training.evaluate_online_hj import MODES
        def record(step,value):
            metrics={mode:dict(collision=value,intervention_fraction=value,mean_speed=15.,
                mean_absolute_gap_error=2.,collision_rate_when_predicted_safe=value,
                missed_collision_fraction=None,predicted_safe_cases=0) for mode in MODES}
            return dict(step=step,modes=metrics,paired_cases=0 if value is None else 6,excluded_cases=0)
        figure=evaluation_figure([record(0,None),record(5000,.5)])
        try:
            points=figure.axes[0].lines[0]
            self.assertEqual(list(points.get_xdata()),[0,5000])
            self.assertTrue(np.isnan(points.get_ydata()[0]))
            self.assertEqual(points.get_ydata()[1],.5)
            self.assertTrue(np.isnan(figure.axes[1].lines[1].get_ydata()).all())
        finally:figure.clear()


if __name__ == '__main__': unittest.main()
