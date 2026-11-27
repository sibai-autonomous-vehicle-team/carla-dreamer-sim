"""Compare continuations after a shared, recorded pre-intervention action prefix."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from analysis.debug_generalist_fallback import action_values
from training.braking_scenario import BrakingScenario
from training.evaluate_online_hj import isolated_evaluation, rollout
from training.train_online_hj import (GeneralistPID, OnlineHJ, append_json,
                                      with_closing_starts, with_route_aware_leader)


class PrefixPolicy:
    def __init__(self, nominal, learner, recorded, switch, strategy):
        self.nominal,self.learner,self.recorded = nominal,learner,recorded
        self.switch,self.strategy,self.profile = switch,strategy,nominal.profile
        self.reset()

    def reset(self):
        self.nominal.reset(); self.tick=0

    def propose(self, observation, dt):
        proposed,gap = self.nominal.propose(observation,dt)
        self.pid_proposal = proposed
        if self.tick < self.switch or self.strategy == 'recorded_source':
            action = self.recorded[min(self.tick,len(self.recorded)-1)]['action']
        elif self.strategy == 'full_brake': action=-3.
        elif self.strategy == 'actor': action=self.learner.action(observation,deterministic=True)
        elif self.strategy == 'filter': action=self.learner.filter_action(observation,proposed)['action']
        else: action=proposed
        self.tick+=1
        return action,gap


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--debug',type=Path,default=Path('training/_local/generalist_fallback_debug_2026-11-26'))
    p.add_argument('--outdir',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,help='Evaluate another checkpoint from the same recorded physical histories')
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--start-server',action='store_true')
    p.add_argument('--port',type=int,default=2800)
    p.add_argument('--source-arm',choices=('pid','actor_filter'),default='pid')
    p.add_argument('--lead-steps',type=int,default=0,help='Branch this many ticks before the first threshold crossing')
    p.add_argument('--arms',nargs='+',choices=('recorded_source','full_brake','actor','pid','filter'),
                   default=['recorded_source','full_brake','actor','pid'])
    cli=p.parse_args()
    if cli.repeats<1 or cli.lead_steps<0:p.error('Positive repeat count and nonnegative lead steps required')
    metadata=json.loads((cli.debug/'manifest.json').read_text())
    source=Path(metadata['source_comparison'])
    checkpoint=cli.checkpoint or Path(metadata['checkpoint'])
    learner=OnlineHJ.from_checkpoint(checkpoint,'cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    args=SimpleNamespace(**vars(learner.args))
    args.outdir,args.port,args.server=cli.outdir,cli.port,Path(args.server)
    cli.outdir.mkdir(parents=True,exist_ok=False)
    (cli.outdir/'manifest.json').write_text(json.dumps(dict(source_arm=cli.source_arm,
        source_debug=str(cli.debug),checkpoint=str(checkpoint.resolve()),
        lead_steps=cli.lead_steps,repeats=cli.repeats,arms=cli.arms),indent=2)+'\n')
    nominal=GeneralistPID(args.pid_checkpoint,learner.device,target_gap=args.target_gap,
                          kp=args.pid_kp,ki=args.pid_ki,kd=args.pid_kd)
    import carla
    from training.collect_braking import start_server
    from collect_parallel import stop
    base=with_route_aware_leader(BrakingScenario,args.leader_lateral_accel,args.leader_deceleration)
    scenario_class=with_closing_starts(base,args.leader_lateral_accel,args.leader_deceleration)
    process,world,settings=None,None,None
    results=[]
    try:
        if cli.start_server:process=start_server(args)
        client=carla.Client('127.0.0.1',args.port);client.set_timeout(120.)
        world=client.load_world(args.map);settings=world.get_settings()
        with isolated_evaluation(learner):
            for condition,case_id in [('day_cloudy',0),('night_rainy',5)]:
                args.condition=condition
                original=[json.loads(x) for x in (cli.debug/f'{condition}_case{case_id}_{cli.source_arm}.jsonl').read_text().splitlines()]
                switch=max(0,next(r['tick'] for r in original if r['pid_q']<args.switching_threshold)-cli.lead_steps)
                suite=json.loads((source/f'suite_{condition}.json').read_text())
                case=next(c for c in suite if c['case']==case_id)
                for repeat in range(cli.repeats):
                    arms=cli.arms
                    offset=repeat%len(arms)
                    for arm in arms[offset:]+arms[:offset]:
                        policy=PrefixPolicy(nominal,learner,original,switch,arm)
                        rows=[];branch={}
                        def trace(env,observation,row):
                            row['pid_proposal']=policy.pid_proposal
                            rows.append(row)
                            if row['tick']==switch:
                                actor=learner.action(observation,deterministic=True)
                                q=action_values(learner,observation,[-3.,actor,policy.pid_proposal]).min(0).values.tolist()
                                velocity=env.world.get_snapshot().find(env.nonego.id).get_velocity()
                                branch.update(gap=row['gap'],ego_speed=observation.speed,
                                    leader_speed=(velocity.x**2+velocity.y**2+velocity.z**2)**.5,
                                    full_brake_q=q[0],actor_q=q[1],pid_q=q[2],actor_action=actor,
                                    selected_action=row['action'],time_s=switch*args.dt)
                        result=rollout(learner,policy,client,scenario_class,args,case,'pid',trace_callback=trace)
                        result.update(condition=condition,case=case_id,repeat=repeat,arm=arm,branch=branch)
                        results.append(result)
                        append_json(cli.outdir/'results.jsonl',result)
                        with (cli.outdir/f'{condition}_case{case_id}_{arm}_r{repeat}.jsonl').open('w') as stream:
                            for row in rows:stream.write(json.dumps(row)+'\n')
                        print(json.dumps(result),flush=True)
        summary={}
        for condition in ('day_cloudy','night_rainy'):
            subset=[r for r in results if r['condition']==condition]
            summary[condition]={arm:dict(episodes=sum(r['arm']==arm for r in subset),
                collisions=sum(r['collision'] for r in subset if r['arm']==arm))
                for arm in cli.arms}
            summary[condition]['branch_spread']={key:max(r['branch'][key] for r in subset)-min(r['branch'][key] for r in subset)
                                                 for key in ('gap','ego_speed','leader_speed')}
        (cli.outdir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        (cli.outdir/'.complete').touch()
    finally:
        try:
            if world is not None and settings is not None:world.apply_settings(settings)
        finally:
            if process is not None:stop(process)


if __name__=='__main__':main()
