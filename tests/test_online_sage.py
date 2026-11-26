"""Dual-policy updates, executed-action replay, terminal targets and rollout wiring."""
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
from torch import nn

from training.online_sage import OnlineSAGE, GoalPolicy, following_reward, sac_backup, finalize_performance
from training.train_online_hj import Observation, Transition, Replay, VisualNetwork, finalize_episode


def tiny_encoder(self,history,include_speed,pretrained,freeze_encoder):
    nn.Module.__init__(self)
    self.freeze_batch_norm=True; self.history_frames=history; self.include_speed=include_speed
    self.width=512*history+int(include_speed)
    self.network=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(3,512))
    self.register_buffer('image_mean',torch.zeros(1,3,1,1))
    self.register_buffer('image_std',torch.ones(1,3,1,1))
    if freeze_encoder:self.network.requires_grad_(False)


class SageTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2); torch.manual_seed(42)
        self.args=NS(history_frames=4,include_speed=True,no_pretrained=True,freeze_encoder=True,
            actor_lr=1e-3,critic_lr=1e-3,gamma_start=.85,gamma_final=.9999,alpha=.2,alpha_final=.2,
            anneal_updates=100,tau=.05,performance_gamma=.99,performance_alpha=.2,performance_tau=.005,
            collision_penalty=10.,lane_penalty=5.,safety_objective='collision',switching_threshold=1.)

    def row(self,frame=100):
        image=torch.ones(224,224,3,dtype=torch.uint8)
        first=Observation((image,)*4,(frame,)*4,15.,0)
        second=Observation((image,)*4,(frame,frame,frame,frame+1),14.,0)
        return Transition(first,2.5,second,3.,2.,False,reward=.4)

    def test_sac_target_stops_on_true_terminal_and_keeps_truncation_bootstrap(self):
        result=sac_backup(torch.tensor([1.,1.]),torch.tensor([True,False]),torch.tensor([4.,4.]),
                          torch.tensor([-2.,-2.]),.9,.2)
        torch.testing.assert_close(result,torch.tensor([1.,4.96]))

    def test_reward_prefers_target_gap_and_speed_match_and_smoothness(self):
        self.assertEqual(following_reward(6.,0.,0.,0.,6.),1.)
        self.assertLess(following_reward(20.,0.,0.,0.,6.),0.)
        self.assertLess(following_reward(6.,5.,0.,0.,6.),1.)
        self.assertLess(following_reward(6.,0.,3.,-3.,6.),1.)

    def test_contact_and_lane_terminal_labels_do_not_turn_timeouts_into_failures(self):
        row=self.row(); self.assertFalse(finalize_performance([row],[],[],self.args)[0].terminated)
        row=self.row()
        rows=finalize_episode([row],[(101,2)],[],1.,include_lane_failure=False)
        rows=finalize_performance(rows,[(101,2)],[],self.args)
        self.assertTrue(rows[0].failed); self.assertTrue(rows[0].terminated)
        self.assertAlmostEqual(rows[0].reward,-9.6)
        row=self.row()
        rows=finalize_episode([row],[],[101],1.,include_lane_failure=False)
        rows=finalize_performance(rows,[],[101],self.args)
        self.assertFalse(rows[0].failed); self.assertTrue(rows[0].terminated)
        self.assertAlmostEqual(rows[0].reward,-4.6)

    def test_both_policy_pairs_update_and_roundtrip_as_frozen_evaluator(self):
        with patch.object(VisualNetwork,'__init__',tiny_encoder),TemporaryDirectory() as directory:
            learner=OnlineSAGE(self.args,'cpu')
            replay=Replay(8); replay.extend([self.row()])
            batch=replay.sample(2,'cpu')
            torch.testing.assert_close(batch[2],torch.full((2,1),2.5))
            before={name:[p.detach().clone() for p in getattr(learner,name).parameters()]
                    for name in ('actor','critics','goal_actor','goal_critics')}
            metrics=learner.update(batch)
            for name,old in before.items():
                self.assertTrue(any(not torch.equal(a,b) for a,b in zip(old,getattr(learner,name).parameters())),name)
            self.assertEqual(learner.updates,1)
            self.assertTrue(torch.isfinite(torch.tensor(list(metrics.values()))).all())
            for targets in (learner.targets,learner.goal_targets):
                self.assertFalse(any(p.requires_grad for p in targets.parameters()))
            for critics in (learner.critics,learner.goal_critics):
                self.assertFalse(any(p.requires_grad for c in critics for p in c.network.parameters()))
            obs=self.row().observation
            proposal=GoalPolicy(learner,6.).propose(obs,.1,deterministic=True)[0]
            self.assertTrue(-3<=proposal<=3)
            path=Path(directory)/'sage.pt'; learner.save(path,123)
            loaded=OnlineSAGE.from_checkpoint(path)
            self.assertAlmostEqual(loaded.goal_action(obs,True),proposal)
            self.assertAlmostEqual(loaded.action(obs,True),learner.action(obs,True))
            self.assertFalse(any(p.requires_grad for name in loaded.model_names for p in getattr(loaded,name).parameters()))

    def test_main_sage_collects_goal_actions_rewards_and_updates_without_pid_files(self):
        from training.train_online_sage import main
        world=NS(frame=100,get_settings=lambda:NS(),apply_settings=lambda value:None)
        world.get_snapshot=lambda:NS(frame=world.frame,timestamp=NS(delta_seconds=.1,elapsed_seconds=world.frame*.1),
            find=lambda actor:NS(get_velocity=lambda:NS(x=15.,y=0.,z=0.)))
        actions=[]; batches=[]
        class Env:
            def __init__(self,*args,**kwargs):
                self.world=world; self.events=[]; self.departures=[]; self.leader_events=[]
                self.ego=NS(id=1); self.nonego=NS(id=2)
            def tracker(self,actor):return NS(steering=lambda:0.)
            def get_rgb_observation(self,frame):return torch.ones(224,224,3,dtype=torch.uint8)
            def step(self,action):actions.append(action); world.frame+=1
            def measurements(self):return dict(collision_events=[],lane_departure_frames=[])
            def close(self):pass
        learner=NS(updates=0,score=lambda obs,action:2. if len(actions)==0 else 0.,
                   goal_action=lambda obs,deterministic=False:1.2,action=lambda obs:-3.,
                   save=lambda path,steps:path.touch())
        def update(batch):
            batches.append(batch); learner.updates+=1
            return dict(updates=learner.updates,critic_mse=0.,gamma=.85,alpha=.2)
        learner.update=update
        with TemporaryDirectory() as directory,ExitStack() as stack:
            root=Path(directory)
            argv=['train','--outdir',str(root/'run'),'--steps','2','--episode-steps','1','--random-steps','0',
                  '--challenge-fraction','0','--learning-starts','1','--batch-size','1','--capacity','8',
                  '--eval-interval','0','--cpu','--pid-root',str(root/'nonexistent')]
            stack.enter_context(patch('sys.argv',argv))
            stack.enter_context(patch.dict('sys.modules',{
                'carla':NS(Client=lambda *args:NS(set_timeout=lambda value:None,load_world=lambda name:world)),
                'training.collect_braking':NS(start_server=lambda args:None),'collect_parallel':NS(stop=lambda process:None)}))
            for target,value in [('training.online_sage.OnlineSAGE',lambda *args:learner),
                ('training.train_online_hj.with_route_aware_leader',lambda *args:Env),
                ('training.train_online_hj.with_closing_starts',lambda base,*args:base),
                ('training.train_online_hj.planned_lane_status',lambda *args:dict(inside=True)),
                ('training.evaluate_closed_loop.make_scenarios',lambda *args:[dict(full_route=[[float(i),0.,0.,0.,0.,0.] for i in range(30)])]),
                ('training.closed_loop.read_snapshot',lambda *args:(world.frame,world.frame*.1,None,6.))]:
                stack.enter_context(patch(target,side_effect=value))
            with redirect_stdout(io.StringIO()):main()
            records=[json.loads(line) for line in (root/'run'/'transitions.jsonl').read_text().splitlines()]
            self.assertEqual(actions,[(1.2,0.),(-3.,0.)])
            self.assertEqual([r['behavior']['source'] for r in records],['goal_actor','safety_actor'])
            self.assertEqual(learner.updates,1); self.assertEqual(len(batches[0]),10)
            self.assertGreater(float(batches[0][8][0]),.9)
            manifest=json.loads((root/'run'/'manifest.json').read_text())
            self.assertEqual(manifest['performance_policy'],'sac'); self.assertTrue(manifest['include_speed'])
            self.assertTrue((root/'run'/'checkpoint_final.pt').exists())


if __name__ == '__main__': unittest.main()
