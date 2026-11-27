"""Select one PID jointly for generalist and context-selector systems, then test.

Validation selects gains; a separate test bank is evaluated only after selection.
Both systems share gains, target, critic, threshold, full-brake fallback and starts.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from .braking_scenario import BrakingScenario, InvalidBrakingScenario
from .compare_generalist_specialist import ContextSelectorPID, validate_shared_pid
from .evaluate_online_hj import isolated_evaluation, make_case, rollout
from .train_online_hj import (GeneralistPID, ContextPID, OnlineHJ, append_json,
                              with_closing_starts, with_route_aware_leader)


ARMS=('generalist','contextual_specialist')
CONDITIONS=tuple(f'{t}_{w}' for t in ('day','sunset','night') for w in ('sunny','cloudy','rainy'))
CANDIDATES=tuple(validate_shared_pid(dict(target_gap=6.,kp=kp,ki=ki,kd=kd))
    for kp,ki,kd in ((.5,0.,.5),(.9,.05,.15),(.5,0.,.15),(.9,0.,.5),(.5,0.,1.),(.9,0.,1.)))


def route_hash(case):
    route=[[round(float(v),2) for v in point[:3]] for point in case['scenario']['full_route']]
    return hashlib.sha256(json.dumps(route,separators=(',',':')).encode()).hexdigest()


def common_cases(records, candidates):
    """Every gain/controller pair must have a valid, matched reset for a case."""
    groups=defaultdict(list)
    for row in records:groups[(row['condition'],row['case'])].append(row)
    expected={(c,arm) for c in candidates for arm in ARMS}
    accepted=set();excluded=[]
    for key,rows in groups.items():
        reason=None
        if len(rows)!=len(expected) or {(r['candidate'],r['arm']) for r in rows}!=expected:
            reason='missing or duplicate candidate/controller'
        elif not all(r['valid'] for r in rows):reason='invalid rollout or sensor failure'
        elif any(max(r[f] for r in rows)-min(r[f] for r in rows)>tol
                 for f,tol in [('initial_gap',.3),('initial_ego_speed',.4),('initial_leader_speed',.4)]):
            reason='reset drift'
        if reason:excluded.append(dict(condition=key[0],case=key[1],reason=reason))
        else:accepted.add(key)
    return accepted,excluded


def aggregate(records, conditions, horizon):
    result={}
    for arm in ARMS:
        rows=[r for r in records if r['arm']==arm]
        per_condition={c:[r for r in rows if r['condition']==c] for c in conditions}
        if any(not rows for rows in per_condition.values()):raise ValueError('Every condition needs paired outcomes')
        balanced=lambda key:sum(sum(float(r[key]) for r in values)/len(values)
                                for values in per_condition.values())/len(conditions)
        survivors=[r for r in rows if r['horizon_complete']]
        result[arm]=dict(episodes=len(rows),collisions=sum(r['collision'] for r in rows),
            survivors=len(survivors),collision_rate=balanced('collision'),
            survival_rate=balanced('horizon_complete'),lane_departure_rate=balanced('lane_departure'),
            mean_duration_fraction=balanced('ticks')/horizon,
            tracking_mae=balanced('mean_absolute_gap_error'),
            mean_speed=balanced('mean_speed'),fallback_fraction=balanced('intervention_fraction'))
    return result


def selection_key(metrics):
    """Safety-first minimax compromise; both systems have equal weight."""
    values=[metrics[arm] for arm in ARMS]
    return (max(m['collision_rate'] for m in values),
            sum(m['collision_rate'] for m in values)/2,
            -min(m['survival_rate'] for m in values),
            -sum(m['survival_rate'] for m in values)/2,
            -sum(m['mean_duration_fraction'] for m in values)/2,
            sum(m['tracking_mae'] for m in values)/2)


def rank_validation(records, candidates, conditions, horizon):
    if any(r.get('split')!='validation' for r in records):
        raise ValueError('PID selection accepts validation records only')
    accepted,excluded=common_cases(records,candidates)
    for condition in conditions:
        for challenge in (False,True):
            if not any(r['condition']==condition and r['challenge']==challenge and
                       (r['condition'],r['case']) in accepted for r in records):
                raise ValueError(f'No common {condition} challenge={challenge} validation cases')
    ranking=[]
    for candidate in candidates:
        chosen=[r for r in records if r['candidate']==candidate and (r['condition'],r['case']) in accepted]
        metrics=aggregate(chosen,conditions,horizon)
        ranking.append(dict(candidate=candidate,profile=candidates[candidate],
                            score=list(selection_key(metrics)),metrics=metrics))
    ranking.sort(key=lambda r:(r['score'],r['candidate']))
    return ranking,accepted,excluded


def write_json(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def write_report(path, selected, test, manifest):
    profile=selected['profile']
    lines=['# Shared-PID comparison', '',
        'One PID was selected jointly on validation scenarios, frozen, and used by both systems on a separate test bank.',
        f"Shared gains: Kp={profile['kp']}, Ki={profile['ki']}, Kd={profile['kd']}; target={profile['target_gap']} m.",
        f"Test: {test['paired_cases']} paired cases across {len(manifest['conditions'])} conditions; horizon={manifest['horizon_s']} s.",
        f"Same frozen critic and threshold {manifest['threshold']}; both fallbacks command full braking (-3, brake=1).", '',
        '| System | Collisions | Survived full horizon | Tracking MAE (m) | Mean speed (m/s) | Gate active |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for arm,m in test['metrics'].items():
        lines.append(f"| {arm} | {m['collisions']}/{m['episodes']} | {m['survivors']}/{m['episodes']} | "
                     f"{m['tracking_mae']:.3f} | {m['mean_speed']:.3f} | {m['fallback_fraction']:.1%} |")
    lines+=['', '## Test by condition', '',
            '| Condition | Generalist collisions | Context-selected specialist collisions |',
            '| --- | ---: | ---: |']
    for condition,metrics in test['by_condition'].items():
        cells=[f"{metrics[a]['collisions']}/{metrics[a]['episodes']}" for a in ARMS]
        lines.append('| '+condition+' | '+' | '.join(cells)+' |')
    lines+=['', '## Selection rule', '',
        'Minimize the worse system collision rate, then the equally weighted mean collision rate. Ties prefer higher worst/mean survival,',
        'longer observed duration, then lower tracking error. All weather conditions have equal weight.',
        'The selected profile is the best tested compromise, not a globally optimal PID or necessarily each system’s individual optimum.', '',
        '## Controls and limitations', '',
        'The target, PID code/anti-windup, gains, critic, threshold, full-brake command, steering, starts and leader behavior are shared.',
        'PID state is independent for each system. The selector keeps its existing derivative reset when it switches CNNs.',
        'Tuning and test banks have different seeds and disjoint exact directed-route hashes. Map roads can overlap;',
        'route disjointness from CNN training data is not established.',
        'Both systems see the same cases; reset drift or invalid runs exclude the case for every candidate/system during tuning.',
        'Models and selector remain frozen. Their training histories differ; the selector was fitted under its original PID settings.',
        'This controls PID tuning and evaluates complete systems; it does not isolate CNN architecture or training-data effects.',
        'Tracking MAE is true-gap error from the target, measured until termination; early collisions censor it.',
        'Full-horizon survival is an episode fraction, not a route-completion fraction. Gate activity can include already-full-brake PID requests.',
        'Only validation results choose gains. Test results are not used to retune gains or the threshold.', '']
    path.write_text('\n'.join(lines))


def plot_search(outdir,ranking,test):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(2,1,figsize=(12,8))
    colors=['#347fb2','#d77a24']
    labels=['Generalist','Context-selected specialist']
    x=np.arange(len(ranking))
    for i,arm in enumerate(ARMS):
        axes[0].bar(x+(i-.5)*.34,[r['metrics'][arm]['collision_rate']*100 for r in ranking],
                    width=.33,color=colors[i],label=labels[i])
    axes[0].axvspan(-.48,.48,color='#2c985c',alpha=.12)
    axes[0].set_xticks(x,[f"{r['candidate']}\nKp={r['profile']['kp']} Ki={r['profile']['ki']} Kd={r['profile']['kd']}" for r in ranking],fontsize=8)
    axes[0].set_title('Validation search — selected shared PID is highlighted; candidates sorted by selection rule')
    axes[0].legend(fontsize=9)
    metrics=[test['metrics'][arm] for arm in ARMS]
    axes[1].bar(np.arange(2),[m['collision_rate']*100 for m in metrics],color=colors,width=.5)
    axes[1].set_xticks(np.arange(2),labels)
    for i,m in enumerate(metrics):axes[1].text(i,m['collision_rate']*100+2,f"{m['collisions']}/{m['episodes']} collisions",ha='center')
    axes[1].set_title('Held-out test — the same selected PID and full-brake fallback for both systems')
    for ax in axes:
        ax.set_ylim(0,110);ax.set_ylabel('Collision rate (%)');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.text(.02,.015,'All nine conditions have equal weight. Validation selects gains; test results never retune gains or the switching threshold.',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(outdir/'shared_pid_comparison.png',dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--outdir',type=Path,required=True)
    p.add_argument('--validation-cases',type=int,default=4)
    p.add_argument('--test-cases',type=int,default=6)
    p.add_argument('--validation-seed',type=int,default=7000001)
    p.add_argument('--test-seed',type=int,default=9000007)
    p.add_argument('--port',type=int,default=2800)
    p.add_argument('--start-server',action='store_true')
    p.add_argument('--server',type=Path,default=Path('.local/carla-0.9.15/CarlaUE4.sh'))
    p.add_argument('--selector',type=Path,default=Path('experiments/logistic_context_selector_2026-11-19/model.json'))
    p.add_argument('--context-cnn',type=Path,default=Path('generalist/cnn.pth'))
    p.add_argument('--resume',action='store_true')
    cli=p.parse_args()
    if any(n<2 or n%2 for n in (cli.validation_cases,cli.test_cases)):p.error('Case counts must be even and >= 2')
    if cli.validation_seed==cli.test_seed:p.error('Validation and test seeds must differ')
    if any(not 0<=n<2**32 for n in (cli.validation_seed,cli.test_seed)):p.error('Seeds must fit uint32')
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learner=OnlineHJ.from_checkpoint(cli.checkpoint,device)
    if getattr(learner.args,'offline_backup_policy',None)!='full-brake':
        p.error('Fair run requires a critic trained with full-brake continuation')
    learner.args.performance_policy='pid'
    args=SimpleNamespace(**vars(learner.args))
    args.outdir,args.port,args.server=cli.outdir,cli.port,cli.server
    generalist=GeneralistPID(args.pid_checkpoint,device,**CANDIDATES[0])
    from selector.model import Selector
    selector=Selector.load(cli.selector)
    context=GeneralistPID(cli.context_cnn,device)
    expected=json.loads(cli.context_cnn.with_suffix('.json').read_text())['cnn_sha256']
    if context.sha256!=expected:raise ValueError('Selector context CNN checksum mismatch')
    bank={condition:ContextPID(Path(args.pid_root)/condition,device) for condition in CONDITIONS}
    candidates={f'pid_{i}':profile for i,profile in enumerate(CANDIDATES)}
    config=dict(checkpoint=str(cli.checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(cli.checkpoint.read_bytes()).hexdigest(),
        conditions=list(CONDITIONS),candidates=candidates,validation_cases=cli.validation_cases,test_cases=cli.test_cases,
        validation_seed=cli.validation_seed,test_seed=cli.test_seed,threshold=args.switching_threshold,
        backup='full_brake',critic_backup='full-brake',horizon_s=args.episode_steps*args.dt,
        safety_objective=args.safety_objective,max_separation=args.max_separation,
        cnn_sha256=generalist.sha256,context_cnn_sha256=context.sha256,
        selector_sha256=hashlib.sha256(cli.selector.read_bytes()).hexdigest(),specialists={c:b.sha256 for c,b in bank.items()})
    cli.outdir.mkdir(parents=True,exist_ok=cli.resume)
    if cli.resume:
        if json.loads((cli.outdir/'manifest.json').read_text())!=config:raise ValueError('Resume configuration differs')
    else:write_json(cli.outdir/'manifest.json',config)
    records_path=cli.outdir/'rollouts.jsonl'
    records=[json.loads(x) for x in records_path.read_text().splitlines()] if cli.resume and records_path.exists() else []
    seen={(r['split'],r['condition'],r['case'],r['candidate'],r['arm']) for r in records}
    import carla
    from .collect_braking import start_server
    from collect_parallel import stop
    base=with_route_aware_leader(BrakingScenario,args.leader_lateral_accel,args.leader_deceleration)
    scenario_class=with_closing_starts(base,args.leader_lateral_accel,args.leader_deceleration)
    process,world,settings=None,None,None

    def run(policy,case,mode):
        for attempt in range(3):
            try:return rollout(learner,policy,client,scenario_class,args,deepcopy(case),mode)
            except InvalidBrakingScenario as error:return dict(valid=False,error=str(error))
            except RuntimeError as error:
                if not str(error).startswith(('IMU timeout','Camera timeout','Unusable sensor-frame sequence')):raise
                append_json(cli.outdir/'sensor_retries.jsonl',dict(condition=args.condition,case=case['case'],attempt=attempt,error=str(error)))
                if attempt==2:return dict(valid=False,error=str(error))

    def suite(split,condition,forbidden):
        path=cli.outdir/f'{split}_{condition}_suite.json'
        if path.exists():return json.loads(path.read_text())
        args.eval_seed=cli.validation_seed if split=='validation' else cli.test_seed
        count=cli.validation_cases if split=='validation' else cli.test_cases
        cases=[]
        for index in range(count):
            for attempt in range(30):
                case=make_case(world,args,index,attempt)
                digest=route_hash(case)
                if digest in forbidden:
                    append_json(cli.outdir/'rejected_starts.jsonl',dict(split=split,condition=condition,case=index,attempt=attempt,reason='route overlap'))
                    continue
                baseline=run(generalist,case,'brake')
                if baseline['valid']:
                    case.update(route_sha256=digest,braking_baseline=baseline);cases.append(case);break
                append_json(cli.outdir/'rejected_starts.jsonl',dict(split=split,condition=condition,case=index,attempt=attempt,reason=baseline.get('error','invalid leader')))
            else:raise RuntimeError(f'Cannot construct {split} {condition} case {index}')
        write_json(path,cases)
        return cases

    def evaluate_bank(split,banks,profiles):
        for condition_index,condition in enumerate(CONDITIONS):
            args.condition=condition
            contextual=ContextSelectorPID(selector,context,bank,condition)
            for case in banks[condition]:
                names=list(profiles);offset=(case['case']+condition_index)%len(names)
                for candidate in names[offset:]+names[:offset]:
                    profile=profiles[candidate]
                    generalist.profile=profile.copy();contextual.profile=profile.copy()
                    policies=[('generalist',generalist),('contextual_specialist',contextual)]
                    if (case['case']+condition_index+int(candidate.split('_')[-1]))%2:policies.reverse()
                    for arm,policy in policies:
                        key=(split,condition,case['case'],candidate,arm)
                        if key in seen:continue
                        record=run(policy,case,'filter')
                        record.update(split=split,condition=condition,case=case['case'],candidate=candidate,
                            arm=arm,challenge=case['start']['challenge'],route_sha256=case['route_sha256'])
                        if arm=='contextual_specialist':record['selected_specialists']=dict(contextual.selections)
                        records.append(record);seen.add(key);append_json(records_path,record)
                        print(json.dumps(dict(phase=split,condition=condition,case=case['case'],candidate=candidate,
                            profile=profile,arm=arm,valid=record['valid'],collision=record.get('collision'),
                            completed=sum(r['split']==split for r in records))),flush=True)

    try:
        if cli.start_server:process=start_server(args)
        client=carla.Client('127.0.0.1',cli.port);client.set_timeout(120.)
        world=client.load_world(args.map);settings=world.get_settings()
        client.set_timeout(30.)  # Fail promptly if a running server becomes unresponsive.
        with isolated_evaluation(learner):
            validation={};test={}
            for condition in CONDITIONS:
                args.condition=condition
                validation[condition]=suite('validation',condition,set())
                print('Prepared validation '+condition,flush=True)
            validation_routes={c['route_sha256'] for values in validation.values() for c in values}
            for condition in CONDITIONS:
                args.condition=condition
                test[condition]=suite('test',condition,validation_routes)
                if any(c['route_sha256'] in validation_routes for c in test[condition]):raise ValueError('Validation/test route overlap')
                print('Prepared test '+condition,flush=True)
            evaluate_bank('validation',validation,candidates)
            validation_records=[r for r in records if r['split']=='validation']
            ranking,accepted,excluded=rank_validation(validation_records,candidates,CONDITIONS,args.episode_steps)
            write_json(cli.outdir/'validation_ranking.json',dict(ranking=ranking,paired_cases=len(accepted),excluded=excluded))
            selected=ranking[0]
            best=dict(selected['profile'],candidate=selected['candidate'],selection_split='validation',
                      selection_score=selected['score'],validation_seed=cli.validation_seed)
            best_path=cli.outdir/'best_shared_pid.json'
            if best_path.exists() and json.loads(best_path.read_text())!=best:raise ValueError('Selected profile changed on resume')
            write_json(best_path,best)
            print('Selected shared PID '+json.dumps(best),flush=True)
            evaluate_bank('test',test,{selected['candidate']:selected['profile']})
            test_records=[r for r in records if r['split']=='test']
            paired,rejected=common_cases(test_records,[selected['candidate']])
            kept=[r for r in test_records if (r['condition'],r['case']) in paired]
            result=dict(paired_cases=len(paired),excluded=rejected,metrics=aggregate(kept,CONDITIONS,args.episode_steps),
                        by_condition={c:aggregate([r for r in kept if r['condition']==c],[c],args.episode_steps) for c in CONDITIONS})
            write_json(cli.outdir/'test_results.json',result)
            write_report(cli.outdir/'report.md',selected,result,config)
            plot_search(cli.outdir,ranking,result)
            with (cli.outdir/'test_comparison.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=['system',*next(iter(result['metrics'].values())).keys()]);writer.writeheader()
                for arm,m in result['metrics'].items():writer.writerow(dict(system=arm,**m))
            (cli.outdir/'.complete').touch()
            print(json.dumps(result,indent=2),flush=True)
    finally:
        try:
            if world is not None and settings is not None:world.apply_settings(settings)
        finally:
            if process is not None:stop(process)


if __name__=='__main__':main()
