"""Automatic frozen, paired rollouts for the online visual HJ trainer.

Rollouts measure the learned recovery policy, not optimal recoverability. Finite
rollouts without contact are censored; no infinite-horizon value error is claimed.
"""
from contextlib import contextmanager
from copy import deepcopy
import json
import random
import time
from types import SimpleNamespace

import numpy as np
import torch


MODES = ('pid', 'safety', 'filter', 'pid_then_safety')


@contextmanager
def isolated_evaluation(learner):
    """Do not consume training RNG streams or leave network modes changed."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    networks=[learner.actor,learner.critics,learner.targets]
    if hasattr(learner,'goal_actor'): networks += [learner.goal_actor,learner.goal_critics,learner.goal_targets]
    modules = [module for net in networks for module in net.modules()]
    modes = [module.training for module in modules]
    try:
        for module in modules: module.training = False
        with torch.no_grad(): yield
    finally:
        for module, mode in zip(modules, modes): module.training = mode
        random.setstate(python_state); np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None: torch.cuda.set_rng_state_all(cuda_state)


def make_case(world,args,index,attempt=0):
    from .evaluate_closed_loop import make_scenarios
    from .train_online_hj import sample_start
    seed=args.eval_seed+index*1009+attempt*1000003
    rng = random.Random(seed)
    config = SimpleNamespace(**vars(args))
    config.challenge_fraction = float(index % 2 == 1)
    start = sample_start(config, rng)
    start['exploration_episode'] = False
    scenario = make_scenarios(world, 1, seed, args.episode_steps)[0]
    scenario['leader'] = list(scenario['full_route'][20]); scenario['leader'][2] += .5
    scenario['route'] = scenario['full_route'][20:]
    return dict(case=index, seed=seed, attempt=attempt, start=start,
                scenario=scenario, brake_after=rng.uniform(*args.brake_delay_range))


def make_suite(world,args):
    return [make_case(world,args,index) for index in range(args.eval_cases)]


def prepare_suite(learner,nominal,client,world,scenario_class,args,viewer=None):
    """Validate setups before freezing the suite; never select on learned success."""
    from .braking_scenario import InvalidBrakingScenario
    from .train_online_hj import append_json
    suite=[]
    for original in make_suite(world,args):
        for attempt in range(30):
            case=original if attempt==0 else make_case(world,args,original['case'],attempt)
            try:
                baseline=rollout(learner,nominal,client,scenario_class,args,case,'brake',viewer)
                if not baseline['valid']: raise InvalidBrakingScenario('Invalid leader during full-braking baseline')
                case['braking_baseline']=baseline
                suite.append(case)
                break
            except InvalidBrakingScenario as error:
                append_json(args.outdir/'evaluation_rejected_starts.jsonl',
                            dict(case=case['case'],attempt=attempt,seed=case['seed'],error=str(error)))
        else: raise RuntimeError(f'Cannot construct valid held-out case {original["case"]} after 30 attempts')
    return suite


def rollout(learner, nominal, client, scenario_class, args, case, mode, viewer=None):
    from .train_online_hj import FrameHistory, Transition, margin, gap_failure, planned_lane_status, finalize_episode
    from .closed_loop import read_snapshot
    start = case['start']; gap = start['gap']
    config = SimpleNamespace(condition=args.condition, blueprint='vehicle.audi.a2',
        initial_gap=gap+start['closing_speed']*1.5, initial_speed=start['leader_speed'],
        ego_initial_speed=start['ego_speed'], leader_brake_test=True, leader_brake_after=case['brake_after'],
        leader_brake_duration=args.brake_duration, stationary_leader_test=False,
        graphics='vulkan', camera=True, real_time=False,
        handoff_gap_bounds=[max(.05,gap-.15),gap+.15], handoff_speed_bounds=args.speed_range)
    env = None
    try:
        # Repeat the same seed for every mode and checkpoint; CARLA still need
        # not be bitwise deterministic. Actual starts are logged below.
        random.seed(case['seed']); np.random.seed(case['seed'] % 2**32); torch.manual_seed(case['seed'])
        env = scenario_class(client, config, args.dt, deepcopy(case['scenario']), reload_world=False)
        nominal.reset()
        history = FrameHistory(args.history_frames, case['case'], args.dt)
        observation = history.capture(env)
        first_speed = observation.speed
        _, _, _, first_gap = read_snapshot(env,args.dt)
        first_leader_velocity = env.world.get_snapshot().find(env.nonego.id).get_velocity()
        first_leader_speed = (first_leader_velocity.x**2+first_leader_velocity.y**2+first_leader_velocity.z**2)**.5
        rows, invalid_frame, interventions, speed_sum, gap_error_sum = [], None, 0, 0., 0.
        initial_q, probe_q, action_scan = None, None, None
        commands=[]
        ending = 'time_limit'
        for tick in range(args.episode_steps):
            frame, _, _, gap = read_snapshot(env,args.dt)
            if frame != observation.frame_ids[-1]: raise RuntimeError('Evaluation observation/snapshot mismatch')
            if mode=='brake': nominal_action=-3.
            elif getattr(nominal,'is_goal',False): nominal_action,_=nominal.propose(observation,args.dt,deterministic=True)
            else: nominal_action,_=nominal.propose(observation,args.dt)
            if tick == 0 and mode!='brake': initial_q = learner.score(observation,nominal_action)
            if tick==args.history_frames-1 and mode!='brake':
                action_scan=dict(frame=frame,frame_ids=list(observation.frame_ids),
                    brake_q=learner.score(observation,-3.),coast_q=learner.score(observation,0.),
                    throttle_q=learner.score(observation,3.),safety_action=learner.action(observation,deterministic=True))
            intervened = False
            # A motion-value probe needs four real captures, not reset padding.
            # First gather history under the deterministic recovery policy.
            probe_now = mode == 'pid_then_safety' and tick == args.history_frames-1
            if probe_now: probe_q = learner.score(observation,nominal_action)
            if mode == 'brake': action=-3.
            elif mode == 'pid' or probe_now: action = nominal_action
            elif mode == 'filter':
                decision = learner.filter_action(observation,nominal_action)
                action, intervened = decision['action'], decision['intervened']
            else: action = learner.action(observation,deterministic=True)
            if viewer is not None:
                viewer.show(observation,step=0,action=action,steering=-env.tracker(env.ego).steering(),
                    gap=gap,q=learner.score(observation,action) if mode!='brake' else 0.,updates=learner.updates,
                    phase=f'EVALUATION case {case["case"]}: {mode.replace("pid","goal") if getattr(nominal,"is_goal",False) else mode}',metrics=None,
                    actor_active=intervened,epsilon=args.switching_threshold,
                    controller_label=f'EVAL {(mode.replace("pid","goal") if getattr(nominal,"is_goal",False) else mode).upper()}')
            if env.world.get_snapshot().frame != frame: raise RuntimeError('World advanced during evaluation inference')
            env.step((action,-env.tracker(env.ego).steering()))
            successor = history.capture(env)
            next_frame, _, _, next_gap = read_snapshot(env,args.dt)
            if next_frame != successor.frame_ids[-1]: raise RuntimeError('Evaluation successor mismatch')
            time.sleep(args.collision_grace)
            ego_inside = planned_lane_status(env,env.ego)['inside']
            if not ego_inside and next_frame not in env.departures: env.departures.append(next_frame)
            if not planned_lane_status(env,env.nonego)['inside']: invalid_frame = next_frame
            rows.append(Transition(observation,action,successor,margin(gap,args),margin(next_gap,args),gap_failure(next_gap,args)))
            commands.append(action)
            interventions += int(intervened); speed_sum += observation.speed
            gap_error_sum += abs(gap-nominal.profile['target_gap'])
            observation = successor
            if invalid_frame is not None or env.events or env.departures or rows[-1].failed:
                ending = 'event'; break
            if next_gap >= args.max_separation:
                ending = 'separation_truncation'; break
        events = env.measurements()  # Drain delayed callbacks before scoring.
        lo, hi = rows[0].observation.frame_ids[-1], rows[-1].successor.frame_ids[-1]
        contacts = [(f,other) for f,other in events['collision_events'] if lo <= f <= hi]
        contacts += [(f,env.nonego.id) for f,other in env.leader_events if other == env.ego.id and lo <= f <= hi]
        external = [f for f,other in env.leader_events if other != env.ego.id and lo <= f <= hi]
        bad_frames = external + ([invalid_frame] if invalid_frame is not None else [])
        valid = not bad_frames or any(f <= min(bad_frames) for f,_ in contacts)
        rows = finalize_episode(rows,contacts,events['lane_departure_frames'],args.event_failure_margin,
                                include_lane_failure=args.safety_objective != 'collision')
        departures = [f for f in events['lane_departure_frames'] if lo <= f <= hi]
        collision = bool(contacts)
        return dict(valid=valid,collision=collision,failure=bool(rows[-1].failed),
            lane_departure=bool(departures),ending=ending,ticks=len(rows),
            horizon_complete=not collision and not departures and ending == 'time_limit',
            initial_nominal_q=initial_q,probe_q=probe_q,action_scan=action_scan,
            mean_action=sum(commands)/len(commands),minimum_action=min(commands),maximum_action=max(commands),
            predicted_safe=probe_q >= args.switching_threshold if probe_q is not None else None,
            initial_gap=first_gap,initial_ego_speed=first_speed,initial_leader_speed=first_leader_speed,
            intervention_fraction=interventions/(tick+1),mean_speed=speed_sum/(tick+1),
            mean_absolute_gap_error=gap_error_sum/(tick+1),minimum_margin=min(r.next_margin for r in rows))
    finally:
        if env is not None: env.close()


def summarize(records):
    """Compare only cases where all four modes produced valid rollouts."""
    grouped = {}
    for row in records: grouped.setdefault((row['nominal_condition'],row['case']),[]).append(row)
    paired = set()
    for key, rows in grouped.items():
        if len(rows)!=len(MODES) or not all(r['valid'] for r in rows): continue
        # Reject drift between nominally identical resets rather than attribute
        # different initial conditions to controller improvement.
        if any(max(r[field] for r in rows)-min(r[field] for r in rows)>tolerance
               for field,tolerance in [('initial_gap',.3),('initial_ego_speed',.4),('initial_leader_speed',.4)]): continue
        paired.add(key)
    result = dict(paired_cases=len(paired),paired_case_ids=[list(key) for key in sorted(paired)],
                  excluded_cases=len(grouped)-len(paired),modes={})
    for mode in MODES:
        rows = [r for r in records if r['mode']==mode and (r['nominal_condition'],r['case']) in paired]
        metrics = dict(episodes=len(rows))
        for key in ('collision','failure','lane_departure','horizon_complete','intervention_fraction','mean_speed','mean_absolute_gap_error'):
            metrics[key] = sum(r[key] for r in rows)/len(rows) if rows else None
        if mode == 'pid_then_safety':
            probed = [r for r in rows if r['probe_q'] is not None]
            safe = [r for r in probed if r['predicted_safe']]
            collisions = [r for r in probed if r['collision']]
            missed = sum(r['collision'] for r in safe)
            metrics.update(probed_cases=len(probed),predicted_safe_cases=len(safe),collision_cases=len(collisions),false_safe_collisions=missed,
                pre_probe_collision_cases=sum(r['collision'] for r in rows if r['probe_q'] is None),
                collision_rate_when_predicted_safe=missed/len(safe) if safe else None,
                missed_collision_fraction=missed/len(collisions) if collisions else None)
        result['modes'][mode] = metrics
    return result


def evaluate(learner, nominal_policies, client, world, scenario_class, args, steps, viewer=None,tracker=None):
    from .braking_scenario import InvalidBrakingScenario
    from .train_online_hj import append_json
    started = time.monotonic()
    records = []
    print(f'Evaluating frozen HJ at training step {steps}; no replay or optimizer updates.',flush=True)
    with isolated_evaluation(learner):
        suite_path = args.outdir/'evaluation_suite.json'
        if suite_path.exists():
            suite = json.loads(suite_path.read_text())
            if any('braking_baseline' not in case for case in suite):
                raise RuntimeError('Legacy unvalidated evaluation suite: use a fresh output directory; do not mix evaluation histories')
        else:
            suite = prepare_suite(learner,next(iter(nominal_policies.values())),client,world,scenario_class,args,viewer)
            suite_path.write_text(json.dumps(suite,indent=2)+'\n')
        for name, nominal in nominal_policies.items():
            for case in suite:
                for mode in MODES:
                    try: record = rollout(learner,nominal,client,scenario_class,args,case,mode,viewer)
                    except InvalidBrakingScenario as error:
                        record = dict(valid=False,error=str(error))
                    baseline=case['braking_baseline']
                    record.update(step=steps,nominal_condition=name,case=case['case'],challenge=case['start']['challenge'],mode=mode,
                                  braking_group='survived_horizon' if baseline['horizon_complete'] else
                                                'collision' if baseline['collision'] else 'incomplete')
                    records.append(record)
                    append_json(args.outdir/'evaluation_rollouts.jsonl',record)
        summary = dict(step=steps,updates=learner.updates,seconds=time.monotonic()-started,
            nominal_kind='goal' if hasattr(learner,'goal_actor') else 'pid',
            threshold=args.switching_threshold,**summarize(records),
            interpretation='Finite-horizon empirical learned-policy recovery; collision-free outcomes are censored, not certified safe. No optimal HJ ground truth or calibration MSE.',
            by_profile={name:summarize([r for r in records if r['nominal_condition']==name]) for name in nominal_policies},
            by_start={label:summarize([r for r in records if r['challenge']==challenge]) for label,challenge in [('ordinary',False),('closing',True)]})
        summary['braking_baseline']=dict(cases=len(suite),collisions=sum(c['braking_baseline']['collision'] for c in suite),
            survived_horizon=sum(c['braking_baseline']['horizon_complete'] for c in suite),
            interpretation='Immediate full braking with fixed steering, cached once when validating the suite. Failure does not prove inevitable collision.')
        summary['by_braking']={label:summarize([r for r in records if r['braking_group']==label])
                               for label in ('survived_horizon','collision','incomplete')}
        if not summary['paired_cases']:
            print('Evaluation has no valid matched cases; inspect evaluation_rollouts.jsonl. No improvement claim is possible.',flush=True)
        latest_path = args.outdir/'evaluation_latest.json'
        if latest_path.exists():
            previous = json.loads(latest_path.read_text())
            comparable = bool(summary['paired_cases']) and previous['paired_case_ids']==summary['paired_case_ids']
            summary['change_from_previous'] = dict(step=previous['step'],same_matched_cases=comparable,
                collision_rate_delta={mode:summary['modes'][mode]['collision']-previous['modes'][mode]['collision']
                                      for mode in MODES} if comparable else None,
                interpretation='Negative collision-rate deltas mean fewer observed collisions; small suites are screening checks, not proof of improvement.')
        append_json(args.outdir/'evaluation.jsonl',summary)
        latest_path.write_text(json.dumps(summary,indent=2)+'\n')
        from .plot_online_hj_evaluation import plot_evaluations
        plot_evaluations(args.outdir)
        if tracker is not None:
            tracker.evaluation(summary,args.outdir/'evaluation_plot.png',records)
        mode_metrics = summary['modes']
        print(f'EVAL step={steps} matched={summary["paired_cases"]} excluded={summary["excluded_cases"]} '
              f'collision rates: {summary["nominal_kind"]}={mode_metrics["pid"]["collision"]} '
              f'safety={mode_metrics["safety"]["collision"]} filter={mode_metrics["filter"]["collision"]}; '
              f'false-safe contacts={mode_metrics["pid_then_safety"]["false_safe_collisions"]}/'
              f'{mode_metrics["pid_then_safety"]["predicted_safe_cases"]} predicted-safe probes',flush=True)
        print(json.dumps(dict(evaluation=summary)),flush=True)
    return summary
