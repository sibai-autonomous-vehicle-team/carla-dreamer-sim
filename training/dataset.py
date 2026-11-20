"""Episode splits and numerical examples for the two offline experiments."""

import random
from collections import defaultdict
from math import dist, floor, isfinite

import torch


HJ_FEATURES = (
    "gap_m",
    "relative_forward_m",
    "relative_lateral_m",
    "sin_relative_yaw",
    "cos_relative_yaw",
    "ego_forward_speed_mps",
    "relative_forward_velocity_mps",
    "relative_lateral_velocity_mps",
)


def safety_margin(distance, safety_buffer_m, max_gap_m=None):
    """Metres to the nearer boundary; the legacy risk model uses only the lower bound."""
    lower = distance - safety_buffer_m
    if max_gap_m is None:
        return lower
    if not 0 <= safety_buffer_m < max_gap_m or not isfinite(max_gap_m):
        raise ValueError("Require finite 0 <= minimum gap < maximum gap")
    upper = max_gap_m - distance
    return torch.minimum(lower, upper) if torch.is_tensor(distance) else min(lower, upper)


def episode_slices(data):
    ends = data["lengths"].cumsum(0).tolist()
    return [slice(start, end) for start, end in zip([0] + ends[:-1], ends)]


def start_groups(data, tolerance):
    """Connected components of starts with both vehicles within tolerance metres."""
    if not isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Start-position tolerance must be positive and finite")
    starts = data["lengths"].cumsum(0) - data["lengths"]
    xy = data["states"][starts][:, [0, 1, 6, 7]].tolist()
    parent = list(range(len(xy)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # Neighboring ego cells are candidates. Distance checks on both cars avoid
    # both rounding-boundary leaks and grouping distinct leaders by ego pose alone.
    cells = defaultdict(list)
    for i, pos in enumerate(xy):
        cx, cy = floor(pos[0] / tolerance), floor(pos[1] / tolerance)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in cells.get((cx + dx, cy + dy), ()):
                    if dist(pos[:2], xy[j][:2]) <= tolerance and dist(pos[2:], xy[j][2:]) <= tolerance:
                        first, second = root(i), root(j)
                        parent[max(first, second)] = min(first, second)
        cells[cx, cy].append(i)
    return [root(i) for i in range(len(xy))]


def assign_splits(group_ids, train_fraction, val_fraction, seed):
    if not 0 < train_fraction < train_fraction + val_fraction < 1:
        raise ValueError("Train, validation and test fractions must all be positive")

    groups = defaultdict(list)
    for episode, group_id in enumerate(group_ids):
        groups[group_id].append(episode)
    groups = list(groups.values())
    if len(groups) < 3:
        raise ValueError("Three disjoint splits require at least three spawn groups")
    random.Random(seed).shuffle(groups)

    # Cut between whole groups at the cumulative episode counts closest to the targets.
    ends = [0]
    for group in groups:
        ends.append(ends[-1] + len(group))
    train_end = min(range(1, len(groups) - 1),
                    key=lambda i: abs(ends[i] - ends[-1] * train_fraction))
    train_val_fraction = train_fraction + val_fraction
    val_end = min(range(train_end + 1, len(groups)),
                  key=lambda i: abs(ends[i] - ends[-1] * train_val_fraction))
    parts = {"train": groups[:train_end], "val": groups[train_end:val_end],
             "test": groups[val_end:]}
    return {name: [episode for group in part for episode in group]
            for name, part in parts.items()}


def split_episodes(data, cfg):
    if "splits" in data:
        return data["splits"]
    # Preserve the historical risk experiment's split on its original cache.
    group_ids = [tuple(round(x, cfg.spawn_round_decimals) for x in
                       data["states"][rows.start, [0, 1, 6, 7]].tolist())
                 for rows in episode_slices(data)]
    return assign_splits(group_ids, cfg.train_fraction, cfg.val_fraction, cfg.seed)


def first_failure(margins):
    """Index of the first nonpositive buffer margin, or None if never observed."""
    hits = torch.nonzero(margins <= 0)
    return int(hits[0, 0]) if len(hits) else None


def risk_examples(data, ids, cfg):
    """Raw gap histories and minimum margins, on the distance channel's own clock."""
    history, horizon = cfg.history_steps, cfg.horizon_steps
    if min(history, horizon) < 1:
        raise ValueError("history_steps and horizon_steps must be positive")

    features, targets, conditions = [], [], []
    counts = {"complete": 0, "failed": 0, "censored": 0}
    slices = episode_slices(data)
    for episode in ids:
        gap = data["distances"][slices[episode]]
        margins = safety_margin(gap, cfg.safety_buffer_m)
        failure = first_failure(margins)
        end = len(gap) if failure is None else failure + 1
        if end < history:
            continue

        histories = gap[:end].unfold(0, history, 1)
        if failure is None:
            # A non-failure ending gives no information about the missing future.
            valid = max(0, len(histories) - horizon + 1)
            counts["censored"] += len(histories) - valid
        else:
            # Extend the first crossed margin as an absorbing value; later measurements
            # (including any recovery or deeper penetration) are outside this objective.
            padding = margins[failure].expand(horizon - 1)
            margins = torch.cat((margins[:end], padding))
            valid = len(histories)
        if valid == 0:
            continue

        future = margins[history - 1:].unfold(0, horizon, 1)
        minimum = future.amin(dim=1)
        failed = int((minimum <= 0).sum())
        counts["failed"] += failed
        counts["complete"] += valid - failed
        features.append(histories[:valid])
        targets.append(minimum)
        conditions.extend([data["conditions"][episode]] * valid)

    if not features:
        return torch.empty(0, history), torch.empty(0), conditions, counts
    return torch.cat(features), torch.cat(targets), conditions, counts


def decision_features(previous_pose, current_pose, current_gap, dt):
    """Eight decision-time features for single or batched synchronized observations."""
    if dt <= 0 or not isfinite(dt):
        raise ValueError("Expected a positive finite time step")
    if current_pose.shape != previous_pose.shape or current_pose.shape[-1] != 12:
        raise ValueError("Expected matching previous and current twelve-slot poses")
    gap = torch.as_tensor(current_gap, dtype=current_pose.dtype, device=current_pose.device)
    if gap.shape != current_pose.shape[:-1]:
        raise ValueError("Expected one current clearance per pose")
    angle = torch.deg2rad(current_pose[..., 4])
    forward = torch.stack((angle.cos(), angle.sin()), dim=-1)
    lateral = torch.stack((-angle.sin(), angle.cos()), dim=-1)
    relative_position = current_pose[..., 6:8] - current_pose[..., :2]
    ego_velocity = (current_pose[..., :2] - previous_pose[..., :2]) / dt
    leader_velocity = (current_pose[..., 6:8] - previous_pose[..., 6:8]) / dt
    relative_velocity = leader_velocity - ego_velocity
    relative_yaw = torch.deg2rad(current_pose[..., 10] - current_pose[..., 4])
    return torch.stack((
        gap, (relative_position * forward).sum(-1), (relative_position * lateral).sum(-1),
        relative_yaw.sin(), relative_yaw.cos(), (ego_velocity * forward).sum(-1),
        (relative_velocity * forward).sum(-1), (relative_velocity * lateral).sum(-1),
    ), dim=-1)


def observations(poses, gaps, pose_rows, cfg):
    # Gap row i−1 best matches pose i in these archives. Live callers supply
    # a synchronized gap directly, without introducing an extra frame of delay.
    return decision_features(poses[pose_rows - 1], poses[pose_rows],
                             gaps[pose_rows + cfg.gap_offset], cfg.dt)


def transition_window(length, failure, cfg):
    """Eligible current pose indices [start, stop); the successor is i+1."""
    if cfg.gap_offset > 0:
        raise ValueError("A later gap row cannot be used as a current-state feature")
    start = max(1, -cfg.gap_offset, -cfg.action_offset)
    stop = min(length - 1, length - cfg.action_offset)
    if failure is not None:
        stop = min(stop, failure - cfg.gap_offset)
    return start, stop


def hj_examples(data, ids, cfg):
    """Build interior one-step pairs ending no later than the first buffer crossing."""
    if cfg.gap_offset > 0:
        raise ValueError("A later gap row cannot be used as a current-state feature")
    if cfg.dt <= 0 or cfg.longitudinal_saturation <= 0:
        raise ValueError("dt and longitudinal_saturation must be positive")

    states, actions, next_states, margins, next_margins, failures = [], [], [], [], [], []
    slices = episode_slices(data)
    for episode in ids:
        rows = slices[episode]
        poses, gaps = data["states"][rows], data["distances"][rows]
        raw_actions = data["actions"][rows]
        # The local collector chooses u[i] at pose i and records gap[i] after
        # that action, so gap[i-1] describes pose i.
        failure = first_failure(safety_margin(gaps, cfg.safety_buffer_m, cfg.max_gap_m))
        start, stop = transition_window(len(poses), failure, cfg)
        if start >= stop:
            continue

        pose_rows = torch.arange(start, stop + 1)
        obs = observations(poses, gaps, pose_rows, cfg)
        command = raw_actions[pose_rows[:-1] + cfg.action_offset].clone()
        # Commands above +3 already give full throttle in this collector. Keep units
        # and the archived steering sign (+steer command becomes -steer in CARLA).
        command[:, 0].clamp_(-cfg.longitudinal_saturation, cfg.longitudinal_saturation)
        margin = safety_margin(obs[:, 0], cfg.safety_buffer_m, cfg.max_gap_m)
        states.append(obs[:-1])
        actions.append(command)
        next_states.append(obs[1:])
        margins.append(margin[:-1])
        next_margins.append(margin[1:])
        # Only the measured buffer crossing justifies an absorbing backup. A final
        # pose at an unexplained recording end is still a normal observed successor.
        failures.append(margin[1:] <= 0)

    if not states:
        empty_state = torch.empty(0, len(HJ_FEATURES))
        return (empty_state, torch.empty(0, 2), empty_state.clone(),
                torch.empty(0), torch.empty(0), torch.empty(0, dtype=torch.bool))
    return tuple(torch.cat(parts) for parts in
                 (states, actions, next_states, margins, next_margins, failures))
