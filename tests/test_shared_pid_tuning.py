"""Shared gains, balanced validation selection, common cohorts and test isolation."""
from types import SimpleNamespace as NS
import unittest

from training.compare_generalist_specialist import ContextSelectorPID, validate_shared_pid
from training.train_online_hj import GeneralistPID
from training.tune_shared_pid import ARMS, common_cases, rank_validation, route_hash, selection_key


class SharedPIDTests(unittest.TestCase):
    def records(self):
        records=[]
        for case in range(2):
            for candidate in ('pid_0','pid_1'):
                for arm in ARMS:
                    collision=(arm=='contextual_specialist') if candidate=='pid_0' else (case==0)
                    records.append(dict(split='validation',condition='day_sunny',case=case,
                        candidate=candidate,arm=arm,challenge=bool(case%2),valid=True,
                        collision=collision,lane_departure=False,horizon_complete=not collision,
                        ticks=50 if collision else 100,mean_absolute_gap_error=2.,mean_speed=10.,
                        intervention_fraction=.1,initial_gap=4.,initial_ego_speed=15.,initial_leader_speed=15.))
        return records

    def test_both_pid_implementations_use_same_values_but_independent_state(self):
        profile=validate_shared_pid(dict(target_gap=6,kp=.7,ki=.02,kd=.8))
        gen=GeneralistPID.__new__(GeneralistPID);gen.profile=profile.copy();gen.reset()
        specialist=ContextSelectorPID(NS(),NS(),{},'day_sunny',**profile)
        self.assertEqual(gen.profile,specialist.profile)
        for key in ('kp','ki','kd'):self.assertEqual(getattr(gen.pid,key),getattr(specialist.pid,key))
        self.assertIsNot(gen.pid,specialist.pid)
        gen.pid.update(8.,.1)
        self.assertEqual(specialist.pid.integral,0.)
        specialist.reset()
        self.assertEqual(specialist.pid.kd,.8)

    def test_minimax_selects_compromise_and_rejects_test_feedback(self):
        records=self.records()
        profiles={'pid_0':{},'pid_1':{}}
        ranking,accepted,excluded=rank_validation(records,profiles,['day_sunny'],100)
        self.assertEqual(ranking[0]['candidate'],'pid_1')
        self.assertEqual(len(accepted),2);self.assertFalse(excluded)
        records[0]['split']='test'
        with self.assertRaisesRegex(ValueError,'validation records only'):
            rank_validation(records,profiles,['day_sunny'],100)

    def test_one_invalid_candidate_excludes_case_for_every_candidate_and_controller(self):
        records=self.records();records[0]['valid']=False
        accepted,excluded=common_cases(records,['pid_0','pid_1'])
        self.assertEqual(accepted,{('day_sunny',1)})
        self.assertEqual(len(excluded),1)
        records=self.records();records[0]['initial_gap']+=.31
        self.assertNotIn(('day_sunny',0),common_cases(records,['pid_0','pid_1'])[0])

    def test_condition_rates_are_balanced_instead_of_weighted_by_episode_count(self):
        from training.tune_shared_pid import aggregate
        records=[]
        for arm in ARMS:
            for c,n,hit in [('day_sunny',10,False),('night_rainy',2,True)]:
                for _ in range(n):
                    records.append(dict(arm=arm,condition=c,collision=hit,horizon_complete=not hit,
                        lane_departure=False,ticks=100,mean_absolute_gap_error=2.,mean_speed=10.,intervention_fraction=0.))
        result=aggregate(records,['day_sunny','night_rainy'],100)
        self.assertEqual(result['generalist']['collision_rate'],.5)

    def test_route_groups_ignore_seeds_and_reject_invalid_gain_profiles(self):
        a=dict(seed=1,scenario=dict(full_route=[[0.,0.,0.,0.],[1.,2.,0.,0.]]))
        b=dict(seed=2,scenario=dict(full_route=[[0.,0.,0.,0.],[1.,2.,0.,1.]]))
        self.assertEqual(route_hash(a),route_hash(b))
        b['scenario']['full_route'][1][0]+=1
        self.assertNotEqual(route_hash(a),route_hash(b))
        for field,value in [('kp',True),('ki',-.1),('kd',float('nan')),('target_gap',0.)]:
            profile=dict(target_gap=6.,kp=.5,ki=0.,kd=.5);profile[field]=value
            with self.assertRaises(ValueError):validate_shared_pid(profile)


if __name__=='__main__':unittest.main()
