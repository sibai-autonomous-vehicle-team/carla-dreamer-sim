"""One frozen-policy CARLA rollout on an already reset, synchronous follow scenario."""

import math
from collections import deque

import torch

from .dataset import decision_features


def read_snapshot(env, expected_dt):
    """Both vehicle poses and the front/rear-edge gap come from the same simulator frame."""
    world = env.ego.get_world()
    snapshot = world.get_snapshot()
    if not math.isclose(snapshot.timestamp.delta_seconds, expected_dt, abs_tol=1e-4):
        raise ValueError("Simulator tick differs from the checkpoint timestep")
    transforms = []
    pose = []
    for actor in (env.ego, env.nonego):
        actor_state = snapshot.find(actor.id)
        if actor_state is None:
            raise RuntimeError(f"Vehicle {actor.id} missing from frame {snapshot.frame}")
        transform = actor_state.get_transform()
        transforms.append(transform)
        pos, rot = transform.location, transform.rotation
        pose.extend((pos.x, pos.y, pos.z, rot.pitch, rot.yaw, rot.roll))
    gap = float(env.get_distance(*transforms))
    if not math.isfinite(gap) or gap < 0 or not all(map(math.isfinite, pose)):
        raise ValueError("Invalid measured CARLA state")
    return snapshot.frame, snapshot.timestamp.elapsed_seconds, torch.tensor(pose, dtype=torch.float32), gap


def run_episode(env, policy, nominal_pid, *, mode, scenario_id, seed, condition, max_steps):
    """The evaluation harness resets the scenario, controls the clock, and supplies the braking PID.

    nominal_pid(env) proposes a *requested* action with exploration disabled. Pass
    an unwrapped CarlaWrapper. Start a fresh call and reset PID memory for each mode.
    """
    if mode not in ("pid", "actor_only", "safety_filter") or max_steps < 1:
        raise ValueError("Specify a supported mode and a positive rollout horizon")
    world = env.ego.get_world()
    dt = policy.settings["dt"]
    if (env._config.action.discrete or not world.get_settings().synchronous_mode or
            not math.isclose(world.get_settings().fixed_delta_seconds or 0, dt, abs_tol=1e-5)):
        raise ValueError("The HJ evaluator requires continuous commands and synchronous fixed-delta stepping")
    frame, time_s, pose, gap = read_snapshot(env, dt)
    trace = {
        "scenario_id": scenario_id, "seed": seed, "condition": condition, "mode": mode,
        "initial_frame": frame, "initial_pose": pose.tolist(), "initial_gap_m": gap,
        "leader_id": env.nonego.id, "d_min": policy.settings["safety_buffer_m"],
        "d_max": policy.settings["max_gap_m"], "rows": [], "ending": "horizon",
    }
    previous = None
    history_length = getattr(policy,"history_frames",1)
    rgb_history = deque(maxlen=history_length)
    for _ in range(max_steps):
        rgb_observation = None
        if getattr(policy, "observation_kind", "state") == "rgb":
            image = env.get_rgb_observation(frame)
            if not rgb_history:
                rgb_history.extend([image]*history_length)
            else:
                rgb_history.append(image)
            rgb_observation = image if history_length == 1 else torch.stack(list(rgb_history))
        nominal = tuple(float(value) for value in nominal_pid(env))
        if len(nominal) != 2 or not all(map(math.isfinite, nominal)) or abs(nominal[1]) > 1:
            raise ValueError("PID must supply finite longitudinal and steering requests, |steer| <= 1")
        action, score, intervened, closing = nominal, None, False, None
        if previous is not None:
            state = decision_features(previous, pose, torch.tensor(gap), dt)
            separation = math.hypot(state[1], state[2])
            if separation > 0:
                closing = -float(state[1:3] @ state[6:8]) / separation
            observation = (rgb_observation
                           if getattr(policy, "observation_kind", "state") == "rgb" else state)
            if mode == "safety_filter":
                decision = policy.filter_action(observation, nominal)
                action, score, intervened = decision.action, decision.nominal_q, decision.intervened
            elif mode == "actor_only":
                action = policy.predict_safe_action(observation)
            if (score is None and getattr(env, "wants_hj_score", False)
                    and getattr(policy, "has_safety_critic", True)):
                score = policy.score_action(observation, nominal)
        render = getattr(env, "render_decision", None)
        if render is not None:
            render(score, mode, intervened)
        if world.get_snapshot().frame != frame:
            raise RuntimeError("World advanced before action application; use one synchronous ticking client")
        # CarlaBaseEnv.get_vehicle_control applies the steering sign once, inside step.
        _, _, done, _ = env.step(action)
        next_frame, next_time, next_pose, next_gap = read_snapshot(env, dt)
        if next_frame != frame + 1 or not math.isclose(next_time - time_s, dt, abs_tol=1e-4):
            raise RuntimeError("Nonconsecutive CARLA snapshots; episode timing is unusable")
        trace["rows"].append({
            "frame": frame, "next_frame": next_frame, "gap_m": gap, "next_gap_m": next_gap,
            "closing_speed_mps": closing, "has_history": previous is not None,
            "nominal_action": nominal, "executed_request": action, "nominal_q_m": score,
            "nominal_gap_m": getattr(nominal_pid, "gap_m", None),
            "nominal_image_frame": getattr(nominal_pid, "image_frame", None),
            "intervened": intervened, "ego_travel_m": float((next_pose[:2] - pose[:2]).norm()),
            "leader_travel_m": float((next_pose[6:8] - pose[6:8]).norm()),
        })
        previous, pose, gap, frame, time_s = pose, next_pose, next_gap, next_frame, next_time
        if done:
            trace["ending"] = "environment_terminal"
            break
    return trace


def summarize_episode(trace, *, collision_events=None, lane_departure_frames=None,
                      unnecessary_brake_frames=None, route_progress_m=None):
    """Call after draining the episode's sensors; missing measurements remain None.

    collision_events contains (frame, other_actor_id) from the ego contact sensor.
    lane_departure_frames contains assessor-confirmed departure onsets. Optional
    unnecessary_brake_frames labels executed command frames using a declared criterion.
    """
    rows = trace["rows"]
    start, end = trace["initial_frame"], rows[-1]["next_frame"]
    gaps = [trace["initial_gap_m"], *(row["next_gap_m"] for row in rows)]
    low, high = trace["d_min"], trace["d_max"]
    contacts = None if collision_events is None else [
        other for frame, other in collision_events if start <= frame <= end
    ]
    eligible = sum(row["has_history"] for row in rows)
    interventions = sum(row["intervened"] for row in rows)
    # A brake while the gap is permitted and no longer closing is an observable proxy;
    # confirming an unnecessary brake requires a separate scenario-level assessment.
    clear_nonclosing = sum(row["executed_request"][0] < 0 and low < row["gap_m"] < high
                           and row["closing_speed_mps"] is not None and row["closing_speed_mps"] <= 0
                           for row in rows)
    braking_frames = {row["frame"] for row in rows if row["executed_request"][0] < 0}
    return {
        **{k: trace[k] for k in ("scenario_id", "seed", "condition", "mode", "ending")},
        "steps": len(rows), "history_eligible_steps": eligible,
        "initial_near_violation": gaps[0] <= low, "initial_far_violation": gaps[0] >= high,
        "near_violation": min(gaps) <= low, "far_violation": max(gaps) >= high,
        "min_gap_m": min(gaps), "max_gap_m": max(gaps), "mean_gap_m": sum(gaps) / len(gaps),
        "permitted_gap_fraction": sum(low < d < high for d in gaps) / len(gaps),
        "interventions": interventions if trace["mode"] == "safety_filter" else None,
        "intervention_rate_eligible": (interventions / eligible
                                      if eligible and trace["mode"] == "safety_filter" else None),
        "braking_decisions": len(braking_frames), "brakes_inside_nonclosing": clear_nonclosing,
        "unnecessary_braking_count": (None if unnecessary_brake_frames is None
                                      else len(braking_frames.intersection(unnecessary_brake_frames))),
        "ego_travel_m": sum(row["ego_travel_m"] for row in rows),
        "leader_travel_m": sum(row["leader_travel_m"] for row in rows),
        "route_progress_m": route_progress_m,
        "any_collision": None if contacts is None else bool(contacts),
        "contact_event_messages": None if contacts is None else len(contacts),
        "leader_collision": None if contacts is None else trace["leader_id"] in contacts,
        "lane_departures": (None if lane_departure_frames is None else
                            len({frame for frame in lane_departure_frames if start <= frame <= end})),
    }


def run_matched_scenario(policy, reset_scenario, *, scenario_id, seed, condition, max_steps,
                         collect_measurements=None, modes=("pid", "actor_only", "safety_filter")):
    """Run three modes with the same harness reset and return traces and per-mode outcomes.

    reset_scenario(scenario_id, seed, condition) returns (reset_env, fresh_nominal_pid).
    It restores both vehicles, blueprints, leader route, weather and simulator/TM seeds.
    collect_measurements(env, trace) can drain sensor callbacks and supply arguments to
    summarize_episode before the next reset. The harness owns simulator and sensor cleanup.
    """
    if not modes or len(set(modes)) != len(modes) or any(
            mode not in ("pid", "actor_only", "safety_filter") for mode in modes):
        raise ValueError("Choose distinct supported evaluation modes")
    results = {}
    initial = None
    for mode in modes:
        env, nominal = reset_scenario(scenario_id, seed, condition)
        actors = (env.ego, env.nonego)
        before = env.ego.get_world().get_snapshot()
        velocities = []
        for actor in actors:
            sample = before.find(actor.id)
            if sample is None:
                raise RuntimeError("Reset actor missing from the initial snapshot")
            velocity = sample.get_velocity()
            velocities.extend((velocity.x, velocity.y, velocity.z))
        frame, _, pose, gap = read_snapshot(env, policy.settings["dt"])
        if frame != before.frame or not all(map(math.isfinite, velocities)):
            raise RuntimeError("Unusable initial snapshot after reset")
        blueprints = tuple(actor.type_id for actor in actors)
        physical_start = torch.tensor([*pose.tolist(), gap, *velocities])
        if initial is None:
            initial = (blueprints, physical_start)
        elif blueprints != initial[0] or not torch.allclose(initial[1], physical_start, atol=1e-3, rtol=0.):
            delta = (initial[1] - physical_start).abs()
            raise ValueError("Scenario reset did not reproduce both initial vehicle poses, velocities, gap and blueprints"
                             f" (max pose delta={float(delta[:12].max()):.6f}, gap delta={float(delta[12]):.6f} m,"
                             f" max velocity delta={float(delta[13:].max()):.6f} m/s)")
        trace = run_episode(env, policy, nominal, mode=mode, scenario_id=scenario_id, seed=seed,
                            condition=condition, max_steps=max_steps)
        if trace["initial_frame"] != frame:
            raise RuntimeError("Start frame changed between reset verification and rollout")
        trace.update(initial_velocity_mps=velocities, vehicle_blueprints=blueprints)
        measurements = collect_measurements(env, trace) if collect_measurements is not None else {}
        results[mode] = {"trace": trace, "summary": summarize_episode(trace, **measurements)}
    return results
