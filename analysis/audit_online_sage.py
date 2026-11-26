"""Read-only audit of a SAGE run; needs only the Python standard library.

Usage: python analysis/audit_online_sage.py RUN_DIRECTORY
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import mean


def audit(folder):
    def read(name):
        path = folder / f'{name}.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    config = json.loads((folder / 'manifest.json').read_text())
    episodes, updates = read('episodes'), read('updates')
    progress, evaluations = read('progress'), read('evaluation')
    transitions = read('transitions')
    admitted = [r for r in episodes if r['admitted_to_replay']]
    low, high = -config['event_failure_margin'], config['max_gap']
    result = dict(run=str(folder), architecture=config.get('architecture', 'not recorded'),
                  steps=transitions[-1]['step'] if transitions else 0,
                  episodes=len(episodes), endings=dict(Counter(r['ending'] for r in episodes)),
                  admitted_episodes=len(admitted),
                  safety_bounds=[low, high] if config['safety_objective'] == 'collision' else None)
    if config['safety_objective'] == 'collision':
        result['logged_safety_bound_violations'] = [
            dict(step=r['step'], q_mean=r['q_mean'], target_mean=r['target_mean'])
            for r in updates if not low <= r['q_mean'] <= high or not low <= r['target_mean'] <= high]
    result['windows'] = []
    for start in range(0, result['steps'], 5000):
        stop = start + 5000
        rows = [r for r in transitions if start < r['step'] <= stop]
        metrics = [r for r in updates if start < r['step'] <= stop]
        replay = [r for r in progress if start < r['step'] <= stop and r['replay_rows']]
        fractions = [r['failure_rows_in_replay'] / r['replay_rows'] for r in replay]
        window = dict(start=start, stop=min(stop, result['steps']),
                      action_sources=dict(Counter(r['behavior']['source'] for r in rows)),
                      mean_action=mean(r['action'] for r in rows))
        for key in ('critic_mse', 'q_mean', 'goal_q_mean', 'entropy', 'goal_entropy',
                    'safety_action_gradient', 'goal_action_gradient'):
            values = [r[key] for r in metrics if key in r]
            if values: window[key] = mean(values)
        if fractions:
            window['mean_failure_row_fraction'] = mean(fractions)
            window['mean_probability_batch_has_no_failure_row'] = mean(
                (1 - f)**config['batch_size'] for f in fractions)
        result['windows'].append(window)
    result['evaluation'] = [dict(step=r['step'], paired_cases=r['paired_cases'],
                                 excluded_cases=r['excluded_cases'], modes=r['modes']) for r in evaluations]
    result['challenge_outcomes'] = {}
    for challenge in (False, True):
        rows = [r for r in admitted if r['challenge_start'] == challenge]
        result['challenge_outcomes'][str(challenge)] = dict(
            episodes=len(rows), collisions=sum(bool(r['collision_events']) for r in rows),
            ended_before_unpadded_history=sum(r['rows'] < config['history_frames'] for r in rows),
            mean_rows=mean(r['rows'] for r in rows) if rows else None)
    result['discount_horizons_seconds'] = {
        str(gamma): -config['dt'] / math.log(gamma) for gamma in (.85, .99, .9999)}
    result['maximum_normalized_action_entropy'] = math.log(2)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.run_directory), indent=2, allow_nan=False))
