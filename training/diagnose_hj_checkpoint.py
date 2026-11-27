"""Evaluate a saved HJ actor against greedy twin-critic grid control; no training."""
import argparse
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from . import train_online_hj as hj
from .evaluate_online_hj import isolated_evaluation, prepare_suite, rollout, summarize


class GridController:
    def __init__(self, learner, points=21):
        self.learner, self.points = learner, points

    def __getattr__(self, name):
        return getattr(self.learner, name)

    @torch.no_grad()
    def action(self, observation, deterministic=True):
        images, speed = hj.tensors([observation], self.device)
        grid = torch.linspace(-3., 3., self.points, device=self.device).reshape(-1, 1)
        values = [c.value(c.features(images, speed).expand(self.points, -1), grid)
                  for c in self.critics]
        q = torch.minimum(*values)
        return float(grid[int(q.argmax()), 0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--suite', type=Path, help='Defaults to checkpoint directory/evaluation_suite.json')
    parser.add_argument('--eval-cases', type=int, default=50)
    parser.add_argument('--grid-points', type=int, default=21)
    parser.add_argument('--port', type=int, default=2995)
    parser.add_argument('--server', type=Path, default=Path('.local/carla-0.9.15/CarlaUE4.sh'))
    parser.add_argument('--start-server', action='store_true')
    parser.add_argument('--require-cuda', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    cli = parser.parse_args()
    if cli.eval_cases < 1 or cli.grid_points < 3: parser.error('Positive cases and at least three grid points required')
    if cli.require_cuda and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    payload = torch.load(cli.checkpoint, map_location='cpu', weights_only=True)
    architecture = payload['architecture']
    cls = hj.OnlineHJ
    if architecture == 'online_hj_dinov2_small_cls_v1':
        from .train_dino_hj import DinoHJ
        cls = DinoHJ
    elif architecture != cls.architecture:
        raise ValueError(f'Unsupported architecture: {architecture}')
    if payload['config'].get('performance_policy') != 'safety':
        raise ValueError('This diagnostic requires a single safety-policy checkpoint')
    args = SimpleNamespace(**payload['config'])
    args.outdir, args.port, args.server = cli.outdir, cli.port, cli.server
    args.eval_cases = cli.eval_cases
    del payload
    learner = cls.from_checkpoint(cli.checkpoint, device)
    grid = GridController(learner, cli.grid_points)
    args.outdir.mkdir(parents=True, exist_ok=False)
    (args.outdir/'diagnostic_config.json').write_text(json.dumps(dict(
        checkpoint=str(cli.checkpoint.resolve()),architecture=architecture,
        grid_points=cli.grid_points,training_updates=False,config=vars(args)),default=str,indent=2))
    from .braking_scenario import BrakingScenario, InvalidBrakingScenario
    from .collect_braking import start_server
    from collect_parallel import stop
    import carla
    scenario_class = hj.with_route_aware_leader(BrakingScenario,args.leader_lateral_accel,args.leader_deceleration)
    scenario_class = hj.with_closing_starts(scenario_class,args.leader_lateral_accel,args.leader_deceleration)
    process, run = None, None
    try:
        if cli.wandb:
            import wandb
            run = wandb.init(project='carla-sage',name=args.outdir.name,job_type='checkpoint-diagnostic',
                             config=dict(checkpoint=str(cli.checkpoint),grid_points=cli.grid_points))
        if cli.start_server: process = start_server(args)
        client = carla.Client('127.0.0.1',args.port); client.set_timeout(120.)
        if getattr(args,'road_environment','town') == 'straight':
            from .straight_road import create_world
            from .straight_weather import with_straight_weather
            world = create_world(client,args.straight_road_length,args.outdir/'road.xodr')
            scenario_class = with_straight_weather(scenario_class)
        else: world = client.load_world(args.map)
        nominal = hj.SafetyPolicy(learner,args.target_gap)
        records = []
        with isolated_evaluation(learner):
            suite_path = cli.suite or cli.checkpoint.parent/'evaluation_suite.json'
            if suite_path.exists():
                suite = json.loads(suite_path.read_text())[:cli.eval_cases]
                if len(suite) < cli.eval_cases:
                    print(f'Using all {len(suite)} cases from saved suite (requested {cli.eval_cases}).',flush=True)
            else: suite = prepare_suite(learner,nominal,client,world,scenario_class,args)
            if not suite: raise ValueError('Empty evaluation suite')
            (args.outdir/'evaluation_suite.json').write_text(json.dumps(suite,indent=2))
            for case in suite:
                for mode, controller in [('actor',learner),('grid',grid)]:
                    try:
                        row = rollout(controller,hj.SafetyPolicy(controller,args.target_gap),client,
                                      scenario_class,args,case,'safety')
                    except InvalidBrakingScenario as error: row = dict(valid=False,error=str(error))
                    row.update(case=case['case'],nominal_condition='safety',mode=mode)
                    records.append(row)
                    hj.append_json(args.outdir/'rollouts.jsonl',row)
                    print(json.dumps(row),flush=True)
        summary = summarize(records,required_modes=('actor','grid'))
        summary['modes'] = {k:summary['modes'][k] for k in ('actor','grid')}
        paired = {tuple(key) for key in summary['paired_case_ids']}
        for mode in ('actor','grid'):
            rows = [r for r in records if r['mode']==mode and (r['nominal_condition'],r['case']) in paired]
            failures = [r['first_failure'] for r in rows if r.get('first_failure')]
            summary['modes'][mode].update(first_failure_causes=dict(Counter(
                cause for f in failures for cause in f['causes'])),
                mean_time_to_failure_s=sum(f['seconds'] for f in failures)/len(failures) if failures else None)
        (args.outdir/'summary.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps(summary,indent=2),flush=True)
        if run:
            run.summary.update(summary)
            run.log({f'diagnostic/{mode}/{k}':v for mode,m in summary['modes'].items()
                     for k,v in m.items() if isinstance(v,(int,float))})
    finally:
        if process is not None: stop(process)
        if run is not None: run.finish()


if __name__ == '__main__': main()
