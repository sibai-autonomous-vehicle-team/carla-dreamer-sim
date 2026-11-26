"""Frozen paired CARLA benchmark of generalist HJ fallback and specialist PID.

python -m training.compare_generalist_specialist --start-server \
  --checkpoint training/_local/generalist_hj_run2/checkpoint_interrupted.pt \
  --outdir training/_local/generalist_specialist_comparison
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from .braking_scenario import BrakingScenario, InvalidBrakingScenario
from .evaluate_online_hj import isolated_evaluation, prepare_suite, rollout
from .train_online_hj import (ContextPID, GeneralistPID, OnlineHJ, append_json,
                              with_closing_starts, with_route_aware_leader)


ARMS = ('generalist_pid', 'generalist_filtered', 'contextual_specialist')
METRICS = ('collision', 'lane_departure', 'horizon_complete',
           'mean_absolute_gap_error', 'mean_speed', 'intervention_fraction')


class ContextSelectorPID:
    """Existing logistic selection rule on the benchmark's synchronized images."""
    def __init__(self, selector, context_cnn, bank, condition):
        self.selector, self.context_cnn, self.bank, self.condition = selector,context_cnn,bank,condition
        self.profile = dict(target_gap=6.,kp=.9,ki=.05,kd=.15)
        self.reset()

    def reset(self):
        from run_gap_pid import GapPID
        self.pid = GapPID(6.,.9,.05,.15)
        self.step, self.active, self.switches, self.selections = 0,None,0,defaultdict(int)

    @staticmethod
    def estimate(policy, observation):
        frame = observation.frames[-1][None].to(policy.device)
        gap = float(policy.encoder.network(policy.encoder.preprocess(frame))[0,0])
        if not math.isfinite(gap): raise FloatingPointError('Nonfinite selector CNN gap')
        return gap

    @torch.inference_mode()
    def propose(self, observation, dt):
        if self.step % math.ceil(2./dt-1e-6) == 0:
            probabilities = self.selector.score(self.condition,self.estimate(self.context_cnn,observation))
            selected = max(probabilities,key=probabilities.get)
            if self.active is not None and selected != self.active:
                self.pid.previous_error = None
                self.switches += 1
            self.active = selected
        gap = self.estimate(self.bank[self.active],observation)
        self.selections[self.active] += 1
        self.step += 1
        return self.pid.update(gap,dt),gap


def summarize(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[(row['condition'], row['case'])].append(row)
    accepted, excluded = [], []
    for key, rows in sorted(grouped.items()):
        reason = None
        if len(rows) != len(ARMS) or {r['arm'] for r in rows} != set(ARMS):
            reason = 'missing arm'
        elif not all(r['valid'] for r in rows):
            reason = 'invalid rollout or sensor error'
        elif any(max(r[k] for r in rows)-min(r[k] for r in rows) > tol
                 for k, tol in [('initial_gap', .3), ('initial_ego_speed', .4),
                                ('initial_leader_speed', .4)]):
            reason = 'initial-state drift'
        if reason:
            excluded.append(dict(condition=key[0], case=key[1], reason=reason))
        else:
            accepted.extend(rows)

    def aggregate(rows):
        result = {}
        for arm in ARMS:
            selected = [r for r in rows if r['arm'] == arm]
            result[arm] = dict(episodes=len(selected), collisions=sum(r['collision'] for r in selected))
            result[arm].update({k:sum(r[k] for r in selected)/len(selected) if selected else None
                                for k in METRICS})
        return result

    paired_outcomes = {}
    for baseline in ('generalist_pid','contextual_specialist'):
        pairs = []
        for key in sorted({(r['condition'],r['case']) for r in accepted}):
            pair = {r['arm']:r for r in accepted if (r['condition'],r['case'])==key}
            pairs.append((pair[baseline]['collision'],pair['generalist_filtered']['collision']))
        paired_outcomes[baseline] = dict(
            filtered_only_collision=sum(not base and gated for base,gated in pairs),
            baseline_only_collision=sum(base and not gated for base,gated in pairs),
            both_collision=sum(base and gated for base,gated in pairs),
            neither_collision=sum(not base and not gated for base,gated in pairs))
    return dict(paired_cases=len(accepted)//len(ARMS), excluded=excluded,paired_outcomes=paired_outcomes,
                arms=aggregate(accepted),
                by_condition={c:aggregate([r for r in accepted if r['condition'] == c])
                              for c in sorted({r['condition'] for r in records})},
                by_start={label:aggregate([r for r in accepted if r['challenge'] == challenge])
                          for label, challenge in [('ordinary', False), ('closing', True)]})


def write_report(outdir, result, manifest):
    (outdir/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    with (outdir/'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['arm', 'episodes', 'collisions', *METRICS])
        writer.writeheader()
        for arm, metrics in result['arms'].items():
            writer.writerow(dict(arm=arm, **metrics))
    lines = ['# Generalist fallback versus contextual specialist PID', '',
             f"Frozen checkpoint: {manifest['training_mode']}, {manifest['updates']} updates.",
             f"Conditions: {', '.join(manifest['conditions'])}. Threshold: {manifest['threshold']} m.",
             f"Paired cases: {result['paired_cases']}; excluded cases: {len(result['excluded'])}.", '',
             '| Controller | Collisions | Completion | Gap MAE (m) | Speed (m/s) | Fallback |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for arm, m in result['arms'].items():
        if not m['episodes']:
            lines.append(f'| {arm} | no paired cases | — | — | — | — |')
            continue
        lines.append(f"| {arm} | {m['collisions']}/{m['episodes']} | {m['horizon_complete']:.1%} | "
                     f"{m['mean_absolute_gap_error']:.3f} | {m['mean_speed']:.3f} | {m['intervention_fraction']:.1%} |")
    lines += ['', '## By condition', '',
             '| Condition | Generalist PID collisions | Filtered generalist collisions | Context selector PID collisions |',
              '| --- | ---: | ---: | ---: |']
    for condition, arms in result['by_condition'].items():
        cells = [f"{arms[a]['collisions']}/{arms[a]['episodes']}" for a in ARMS]
        lines.append('| '+condition+' | '+' | '.join(cells)+' |')
    lines += ['', '## Paired collision outcomes', '']
    for baseline, counts in result['paired_outcomes'].items():
        lines.append(f"Against {baseline}: filtered-only collisions {counts['filtered_only_collision']}; "
                     f"baseline-only collisions {counts['baseline_only_collision']}.")
    lines += ['', 'Half of the requested cases are close, fast-closing starts. All arms share the same saved scenarios,',
              'leader behavior, steering, 0.1 s timestep and horizon. Actual starts must agree within',
              '0.3 m gap and 0.4 m/s speed across arms; otherwise the whole case is excluded.',
              'Metrics average over episodes. Gap MAE covers each rollout until termination and can be biased by early collisions.',
              'Completion means surviving the configured horizon without contact, departure or separation truncation.',
              'The specialist arm uses the saved logistic selector: weather/time flags plus its original generalist gap CNN,',
              'argmax specialist selection every 2 s, shared PID (0.9, 0.05, 0.15), and derivative reset on switches.',
              'Its acceptance threshold is diagnostic, matching the existing runner; it does not add a fallback.',
              'All arms use the same synchronized camera, route steering and terminal rules in this benchmark.',
              'Generalist PID gains come from the training checkpoint. This compares configured controllers, not the CNN alone.',
              'Safety is collision-only. A small screening suite and a partially trained actor cannot establish safety.',
              'Fresh sampling seeds do not establish route disjointness from all CNN training data.',
              'Sensor failures retry the same scenario at most three times and are recorded in attempts.jsonl.', '']
    (outdir/'report.md').write_text('\n'.join(lines))


def plot_result(outdir, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    conditions = list(result['by_condition'])
    x = np.arange(len(conditions))
    fig, axes = plt.subplots(2,1,figsize=(13,8),sharex=True)
    labels = ('Generalist PID','Generalist + critic/actor fallback','Context selector + specialist PID')
    colors = ('#377eb8','#e68613','#359b73')
    for ax,metric,scale,ylabel in [(axes[0],'collision',100,'Collision rate (%)'),
                                   (axes[1],'mean_absolute_gap_error',1,'Gap MAE (m)')]:
        for i,(arm,label,color) in enumerate(zip(ARMS,labels,colors)):
            values = [result['by_condition'][c][arm][metric] for c in conditions]
            ax.bar(x+(i-1)*.25,[v*scale if v is not None else float('nan') for v in values],
                   width=.24,label=label,color=color)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_ylim(0,105)
    axes[0].legend(loc='upper right',fontsize=9)
    axes[1].set_xticks(x,[c.replace('_',' ')+'\n(n='+str(result['by_condition'][c][ARMS[0]]['episodes'])+')'
                           for c in conditions],rotation=25,ha='right')
    fig.suptitle('Paired lead-braking comparison — lower values are better')
    fig.text(.02,.015,'n = paired cases per condition. Gap errors cover each rollout until termination; early collisions can bias this metric.',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.95))
    fig.savefig(outdir/'comparison.png',dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--conditions', nargs='+', help='Defaults to the checkpoint training conditions')
    parser.add_argument('--cases', type=int, default=12, help='Per condition, half ordinary and half closing')
    parser.add_argument('--seed', type=int, default=3000007)
    parser.add_argument('--port', type=int, default=2800)
    parser.add_argument('--start-server', action='store_true')
    parser.add_argument('--server', type=Path, default=Path('.local/carla-0.9.15/CarlaUE4.sh'))
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--selector',type=Path,default=Path('experiments/logistic_context_selector_2026-11-19/model.json'))
    parser.add_argument('--context-cnn',type=Path,default=Path('generalist/cnn.pth'))
    cli = parser.parse_args()
    conditions = [f'{t}_{w}' for t in ('day','sunset','night') for w in ('sunny','cloudy','rainy')]
    if cli.conditions is not None and (len(set(cli.conditions)) != len(cli.conditions) or any(c not in conditions for c in cli.conditions)):
        parser.error('Conditions must be unique supported weather/lighting names')
    if cli.cases < 2 or cli.cases % 2:
        parser.error('--cases must be a positive even number >= 2')
    if not 0 <= cli.seed < 2**32:
        parser.error('--seed must be in [0,2**32)')
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() and not cli.cpu else 'cpu')
    learner = OnlineHJ.from_checkpoint(cli.checkpoint, device)
    args = SimpleNamespace(**vars(learner.args))
    cli.conditions = cli.conditions or getattr(args,'offline_conditions',[args.condition])
    args.outdir, args.port, args.server = cli.outdir, cli.port, cli.server
    args.eval_cases, args.eval_seed = cli.cases, cli.seed
    generalist = GeneralistPID(args.pid_checkpoint, device, target_gap=args.target_gap,
                              kp=args.pid_kp, ki=args.pid_ki, kd=args.pid_kd)
    from selector.model import Selector
    selector = Selector.load(cli.selector)
    context_cnn = GeneralistPID(cli.context_cnn,device)
    expected = json.loads(cli.context_cnn.with_suffix('.json').read_text())['cnn_sha256']
    if expected != context_cnn.sha256: raise ValueError('Context CNN checksum mismatch')
    specialists = {c:ContextPID(Path(args.pid_root)/c, device) for c in conditions}
    if any(p.profile['target_gap'] != args.target_gap for p in specialists.values()):
        raise ValueError('Specialist and generalist target gaps must match')
    args.outdir.mkdir(parents=True, exist_ok=False)
    manifest = dict(checkpoint=str(cli.checkpoint.resolve()),
                    checkpoint_sha256=hashlib.sha256(cli.checkpoint.read_bytes()).hexdigest(),
                    steps=0, updates=learner.updates, conditions=cli.conditions,
                    training_mode='offline pooled nine-condition HJ-SAC' if hasattr(args,'offline_conditions') else 'online HJ-SAC',
                    threshold=args.switching_threshold, cases_per_condition=cli.cases,
                    seed=cli.seed, dt=args.dt, horizon=args.episode_steps,
                    generalist=dict(sha256=generalist.sha256, profile=generalist.profile),
                    selector=dict(path=str(cli.selector.resolve()),sha256=hashlib.sha256(cli.selector.read_bytes()).hexdigest(),
                                  context_cnn_sha256=context_cnn.sha256,decision_interval_s=2.,pid=[.9,.05,.15]),
                    specialists={c:dict(sha256=p.sha256, profile=p.profile) for c,p in specialists.items()})
    # The model loader retains update count; steps are metadata only.
    payload = torch.load(cli.checkpoint, map_location='cpu', weights_only=True)
    manifest['steps'] = payload['steps']
    del payload
    (args.outdir/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    import carla
    from .collect_braking import start_server
    from collect_parallel import stop
    process, world, original = None, None, None
    records = []
    base = with_route_aware_leader(BrakingScenario,args.leader_lateral_accel,args.leader_deceleration)
    scenario_class = with_closing_starts(base,args.leader_lateral_accel,args.leader_deceleration)

    def run(policy, case, mode):
        for attempt in range(3):
            try:
                return rollout(learner,policy,client,scenario_class,args,case,mode)
            except RuntimeError as error:
                if not str(error).startswith(('IMU timeout', 'Camera timeout', 'Unusable sensor-frame sequence')):
                    raise
                append_json(args.outdir/'attempts.jsonl',dict(condition=args.condition,case=case['case'],
                            mode=mode,attempt=attempt,error=str(error)))
                print(f"Sensor retry: {args.condition} case={case['case']} mode={mode} attempt={attempt+1}",flush=True)
                if attempt == 2:
                    return dict(valid=False, error=str(error))

    try:
        if cli.start_server: process = start_server(args)
        client = carla.Client('127.0.0.1',args.port)
        client.set_timeout(120.)
        world = client.load_world(args.map)
        original = world.get_settings()
        with isolated_evaluation(learner):
            for condition in cli.conditions:
                args.condition = condition
                contextual = ContextSelectorPID(selector,context_cnn,specialists,condition)
                # Suite creation validates leader setups with a full-braking arm;
                # no controller's success is used to select cases.
                suite = prepare_suite(learner,generalist,client,world,scenario_class,args,
                                      baseline_runner=lambda case:run(generalist,case,'brake'))
                (args.outdir/f'suite_{condition}.json').write_text(json.dumps(suite,indent=2)+'\n')
                for case in suite:
                    arms = [('generalist_pid',generalist,'pid'),
                            ('generalist_filtered',generalist,'filter'),
                            ('contextual_specialist',contextual,'pid')]
                    # Rotate execution order to distribute server drift across arms.
                    offset = case['case'] % len(arms)
                    for arm, policy, mode in arms[offset:]+arms[:offset]:
                        try:
                            record = run(policy,deepcopy(case),mode)
                        except InvalidBrakingScenario as error:
                            record = dict(valid=False,error=str(error))
                        record.update(condition=condition,case=case['case'],arm=arm,
                                      challenge=case['start']['challenge'])
                        if arm == 'contextual_specialist':
                            record.update(selected_specialists=dict(contextual.selections),selector_switches=contextual.switches)
                        records.append(record)
                        append_json(args.outdir/'rollouts.jsonl',record)
                        print(json.dumps(record),flush=True)
                    result = summarize(records)
                    write_report(args.outdir,result,manifest)
        plot_result(args.outdir,result)
        (args.outdir/'.complete').touch()
        print(json.dumps(result,indent=2),flush=True)
    finally:
        try:
            if world is not None and original is not None: world.apply_settings(original)
        finally:
            if process is not None: stop(process)


if __name__ == '__main__':
    main()
