"""Replay the paired fallback regressions with action/value traces and ablations."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from training.braking_scenario import BrakingScenario
from training.evaluate_online_hj import isolated_evaluation, rollout
from training.train_online_hj import (GeneralistPID, OnlineHJ, append_json, tensors,
                                      with_closing_starts, with_route_aware_leader)


class FallbackAblation:
    def __init__(self, learner, strategy):
        self.learner, self.strategy = learner,strategy

    def __getattr__(self, name):
        return getattr(self.learner,name)

    def filter_action(self, observation, nominal):
        decision = self.learner.filter_action(observation,nominal)
        if decision['intervened']:
            if self.strategy == 'brake_filter': decision['action'] = -3.
            elif self.strategy == 'clamped_actor_filter':
                decision['action'] = min(nominal,decision['action'])
        return decision


@torch.no_grad()
def action_values(learner, observation, commands):
    images,speed = tensors([observation],learner.device)
    commands = torch.tensor(commands,device=learner.device,dtype=torch.float32)[:,None]
    values = []
    for critic in learner.critics:
        features = critic.features(images,speed).expand(len(commands),-1)
        values.append(critic.value(features,commands))
    return torch.stack(values).cpu()


def analyze_trace(rows, threshold):
    interventions = [r for r in rows if r['intervened']]
    return dict(ticks=len(rows),interventions=len(interventions),
        first_intervention=interventions[0] if interventions else None,
        actor_weaker_braking=sum(r['actor_action']>r['nominal_action']+.01 for r in interventions),
        actor_accelerates=sum(r['actor_action']>0 for r in interventions),
        actor_q_below_threshold=sum(r['actor_q']<threshold for r in interventions),
        actor_q_below_pid=sum(r['actor_q']<r['pid_q'] for r in interventions),
        tail=rows[-5:])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison',type=Path,default=Path('training/_local/generalist_context_comparison_9conditions_2026-11-26'))
    p.add_argument('--outdir',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,help='Evaluate a corrected checkpoint on the same regression cases')
    p.add_argument('--arms',nargs='+',choices=('pid','actor_filter','brake_filter','clamped_actor_filter'),
                   default=['pid','actor_filter','brake_filter','clamped_actor_filter'])
    p.add_argument('--start-server',action='store_true')
    p.add_argument('--port',type=int,default=2800)
    p.add_argument('--limit',type=int)
    cli = p.parse_args()
    manifest = json.loads((cli.comparison/'manifest.json').read_text())
    original = [json.loads(x) for x in (cli.comparison/'rollouts.jsonl').read_text().splitlines()]
    grouped = defaultdict(dict)
    for row in original: grouped[(row['condition'],row['case'])][row['arm']] = row
    regressions = [key for key,d in grouped.items()
                   if not d['generalist_pid']['collision'] and d['generalist_filtered']['collision']]
    if cli.limit: regressions = regressions[:cli.limit]
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = cli.checkpoint or Path(manifest['checkpoint'])
    learner = OnlineHJ.from_checkpoint(checkpoint,device)
    args = SimpleNamespace(**vars(learner.args))
    args.outdir,args.port,args.server = cli.outdir,cli.port,Path(args.server)
    nominal = GeneralistPID(args.pid_checkpoint,device,target_gap=args.target_gap,
                            kp=args.pid_kp,ki=args.pid_ki,kd=args.pid_kd)
    cli.outdir.mkdir(parents=True,exist_ok=False)
    (cli.outdir/'manifest.json').write_text(json.dumps(dict(source_comparison=str(cli.comparison),
        checkpoint=str(checkpoint.resolve()),regressions=regressions,
        ablations=cli.arms),indent=2)+'\n')
    import carla
    from training.collect_braking import start_server
    from collect_parallel import stop
    base = with_route_aware_leader(BrakingScenario,args.leader_lateral_accel,args.leader_deceleration)
    scenario_class = with_closing_starts(base,args.leader_lateral_accel,args.leader_deceleration)
    process,world,settings = None,None,None
    summaries = []
    try:
        if cli.start_server: process = start_server(args)
        client = carla.Client('127.0.0.1',args.port); client.set_timeout(120.)
        world = client.load_world(args.map); settings = world.get_settings()
        with isolated_evaluation(learner):
            for condition,case_id in regressions:
                args.condition = condition
                suite = json.loads((cli.comparison/f'suite_{condition}.json').read_text())
                case = next(c for c in suite if c['case']==case_id)
                for arm in cli.arms:
                    rows = []
                    def trace(env,observation,record):
                        actor = learner.action(observation,deterministic=True)
                        commands = [record['nominal_action'],actor,-3.,0.,3.]
                        values = action_values(learner,observation,commands)
                        minimum = values.min(0).values.tolist()
                        snap = env.world.get_snapshot()
                        velocity = snap.find(env.nonego.id).get_velocity()
                        leader_speed = (velocity.x**2+velocity.y**2+velocity.z**2)**.5
                        record.update(time_s=record['tick']*args.dt,ego_speed=observation.speed,
                                      leader_speed=leader_speed,closing_speed=observation.speed-leader_speed,
                                      actor_action=actor,pid_q=minimum[0],actor_q=minimum[1],
                                      brake_q=minimum[2],coast_q=minimum[3],throttle_q=minimum[4],
                                      twin_disagreement=(values[0]-values[1]).abs().tolist())
                        rows.append(record)
                    for attempt in range(3):
                        rows.clear()
                        try:
                            result = rollout(FallbackAblation(learner,arm),nominal,client,scenario_class,args,
                                             case,'pid' if arm=='pid' else 'filter',trace_callback=trace)
                            break
                        except RuntimeError as error:
                            if not str(error).startswith(('IMU timeout','Camera timeout','Unusable sensor-frame sequence')) or attempt==2:
                                raise
                    prefix = f'{condition}_case{case_id}_{arm}'
                    with (cli.outdir/f'{prefix}.jsonl').open('w') as stream:
                        for row in rows: stream.write(json.dumps(row)+'\n')
                    result.update(condition=condition,case=case_id,arm=arm,
                                  trace=analyze_trace(rows,args.switching_threshold))
                    summaries.append(result)
                    append_json(cli.outdir/'results.jsonl',result)
                    print(json.dumps(dict(condition=condition,case=case_id,arm=arm,
                                          collision=result['collision'],ticks=result['ticks'],
                                          trace=result['trace']['first_intervention'])),flush=True)
        summary = {arm:dict(episodes=sum(r['arm']==arm for r in summaries),
                          collisions=sum(r['collision'] for r in summaries if r['arm']==arm))
                   for arm in cli.arms}
        (cli.outdir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        (cli.outdir/'.complete').touch()
        print(json.dumps(summary,indent=2),flush=True)
    finally:
        try:
            if world is not None and settings is not None: world.apply_settings(settings)
        finally:
            if process is not None: stop(process)


if __name__ == '__main__':
    main()
