"""Check held-out CARLA setups with immediate full braking, without neural training.

Run on a dedicated idle CARLA server:
python -m training.check_sage_scenarios --outdir training/_local/brake_check --port 2800
Add --start-server --server PATH/CarlaUE4.sh to own a separate server.
This validates physical rollouts and frame matching. A braking collision is not
proof that collision is unavoidable under every policy.
"""
import json
import time
from types import SimpleNamespace

from .train_online_hj import parse_args, with_route_aware_leader, with_closing_starts


def main():
    args=parse_args(dict(performance_policy='sac'),description=__doc__)
    args.outdir.mkdir(parents=True,exist_ok=False)
    import carla
    from .braking_scenario import BrakingScenario
    from .collect_braking import start_server
    from .evaluate_online_hj import prepare_suite
    from collect_parallel import stop
    process=world=original=None
    started=time.monotonic()
    try:
        if args.start_server: process=start_server(args)
        client=carla.Client('127.0.0.1',args.port); client.set_timeout(120.)
        world=client.load_world(args.map); original=world.get_settings()
        scenario_class=with_closing_starts(
            with_route_aware_leader(BrakingScenario,args.leader_lateral_accel,args.leader_deceleration),
            args.leader_lateral_accel,args.leader_deceleration)
        suite=prepare_suite(SimpleNamespace(updates=0),SimpleNamespace(profile=dict(target_gap=args.target_gap)),
                            client,world,scenario_class,args)
        (args.outdir/'evaluation_suite.json').write_text(json.dumps(suite,indent=2)+'\n')
        rows=[]
        for case in suite:
            baseline=case['braking_baseline']
            row=dict(case=case['case'],challenge=case['start']['challenge'],
                gap=baseline['initial_gap'],ego_speed=baseline['initial_ego_speed'],
                leader_speed=baseline['initial_leader_speed'],collision=baseline['collision'],
                horizon_complete=baseline['horizon_complete'],ticks=baseline['ticks'])
            rows.append(row); print(json.dumps(row),flush=True)
        report=dict(seconds=time.monotonic()-started,cases=rows,
            note='Full braking only. No neural policy, optimizer updates, or training replay. Surviving a finite horizon is not a safety guarantee.')
        (args.outdir/'braking_baseline.json').write_text(json.dumps(report,indent=2)+'\n')
    finally:
        if world is not None and original is not None: world.apply_settings(original)
        if process is not None: stop(process)


if __name__=='__main__': main()
