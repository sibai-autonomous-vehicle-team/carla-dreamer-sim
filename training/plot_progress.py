"""Plot saved training history and checkpoint evaluations without retraining.

Example: python -m training.plot_progress --run-dir training/_local/day_sunny_s42
"""
import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / '_local' / 'mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


BLUE, ORANGE = '#2676b8', '#d66c28'


def band(ax, steps, records, key, label, low='p10', high='p90', color=BLUE):
    values = [r[key] for r in records]
    ax.plot(steps, [v['median'] for v in values], color=color, label=label)
    ax.fill_between(steps, [v[low] for v in values], [v[high] for v in values],
                    color=color, alpha=.16)


def save(fig, axes, output, title, subtitle):
    fig.suptitle(title, fontsize=17, x=.07, ha='left')
    fig.text(.07, .925, subtitle, fontsize=10, color='#4b5563')
    for ax in axes.flat:
        ax.set_xlabel('Training updates')
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
        ax.ticklabel_format(axis='x', style='plain', useOffset=False)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, .90))
    fig.savefig(output, dpi=160, facecolor='white')
    plt.close(fig)
    print(output)


def plot_history(metrics, output):
    rows = sorted(metrics['history'], key=lambda r: r['step'])
    if not rows:
        raise ValueError('metrics.json has no training history')
    steps = [r['step'] for r in rows]
    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    ax = axes[0, 0]
    if metrics.get('algorithm') == 'bc':
        ax.plot(steps,[r.get('validation_bc_mse',float('nan')) for r in rows],color=BLUE)
        ax.set(title='Validation controller imitation error',ylabel='Normalized action MSE')
    else:
        ax.plot(steps, [r['critic_mse'] for r in rows], color=BLUE)
        ax.set(title='Critic fitting error against moving training targets', ylabel='Mean squared error', yscale='log')
    ax = axes[0, 1]
    ax.plot(steps, [r['actor_loss'] for r in rows], color=ORANGE)
    algorithm = metrics.get('algorithm','ddpg')
    ax.set(title='Actor objective',ylabel=('Normalized imitation MSE' if algorithm=='bc'
           else 'Imitation + scaled Q loss' if algorithm=='hj-td3-bc' else 'Actor loss = −mean Q'))
    ax = axes[1, 0]
    ax.plot(steps, [100*r['braking_fraction'] for r in rows], color=BLUE)
    ax.set(title='Actor braking on fixed validation probes', ylabel='States with negative request (%)', ylim=(0, 100))
    ax = axes[1, 1]
    if all('q_actor_m' in r for r in rows):
        band(ax, steps, rows, 'q_actor_m', 'Actor action', color=BLUE)
        band(ax, steps, rows, 'q_logged_m', 'Recorded action', color=ORANGE)
    else:
        ax.text(.5,.5,'Supervised baseline: no safety critic',transform=ax.transAxes,ha='center')
    ax.set(title='Validation critic predictions: median and 10–90% range', ylabel='Predicted discounted margin (m)')
    for ax, key, label, limits in [(axes[2, 0], 'longitudinal', 'Longitudinal request', (-3.15, 3.15)),
                                    (axes[2, 1], 'steering', 'Steering request', (-1.05, 1.05))]:
        band(ax, steps, rows, key, 'Actor median; shaded 10–90%')
        ax.set(title=f'Validation {label.lower()}', ylabel=label, ylim=limits)
        ax.axhline(0, color='gray', linewidth=.8)
    fig.text(.07, .012, 'Training losses are not collision rates. Validation probes are recorded states, not closed-loop rollouts.', fontsize=10)
    save(fig, axes, output / 'training_progress.png', 'Training progress',
         f"{metrics.get('device_name', metrics.get('device', ''))} • {metrics.get('validation_probe_count', '?')} fixed validation probes")
    if metrics.get('algorithm') in ('bc','hj-td3-bc'):
        fig, axes = plt.subplots(1,3,figsize=(14,4))
        for ax,key,label in zip(axes,('bc_loss','validation_bc_mse','critic_disagreement'),
                               ('Actor imitation term','Validation imitation error','Twin-critic disagreement')):
            values=[r.get(key,float('nan')) for r in rows]
            if any(v==v for v in values):
                ax.plot(steps,values,color=BLUE)
            else:
                ax.text(.5,.5,'Not applicable',transform=ax.transAxes,ha='center')
            ax.set(title=label,ylabel='Normalized MSE' if key!='critic_disagreement' else 'Predicted margin difference (m)')
        save(fig,axes,output/'algorithm_diagnostics.png','Learning diagnostics',metrics['algorithm'])


def plot_evaluations(metrics, paths, output):
    reports = [json.loads(p.read_text()) for p in paths]
    for r in reports:
        if (r['dataset_id'], r['split_id']) != (metrics['dataset_id'], metrics['split_id']):
            raise ValueError('Evaluation dataset/split identity differs from the training run')
    reports.sort(key=lambda r: r['step'])
    if len({r['step'] for r in reports}) != len(reports):
        raise ValueError('Duplicate checkpoint evaluations for the selected split')
    steps = [r['step'] for r in reports]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    keys = [('brake_fraction', 'Braking requests'),
            ('abs_steer_ge_095_fraction', 'Near-limit steering: |request| ≥ 0.95'),
            ('abs_long_ge_285_fraction', 'Near-limit longitudinal: |request| ≥ 2.85')]
    for ax, (key, title) in zip(axes[0], keys):
        for source, label, color in [('actor', 'Actor', BLUE), ('recorded', 'Recorded controller', ORANGE)]:
            ax.plot(steps, [100*r[source][key] for r in reports], marker='o', label=label, color=color)
        ax.set(title=title, ylabel='Evaluated states (%)', ylim=(0, 100))
    for ax, key, label in [(axes[1, 0], 'longitudinal', 'Longitudinal request'),
                           (axes[1, 1], 'steering', 'Steering request')]:
        for source, legend, color in [('actor', 'Actor', BLUE), ('recorded', 'Recorded controller', ORANGE)]:
            band(ax, steps, [r[source] for r in reports], key, legend, 'p05', 'p95', color)
        ax.set(title=f'{label}: median and 5–95% range', ylabel=label)
        ax.axhline(0, color='gray', linewidth=.8)
    ax = axes[1, 2]
    if all('q_actor_m' in r for r in reports):
        for key, label, color in [('q_actor_m', 'Actor action', BLUE), ('q_recorded_m', 'Recorded action', ORANGE)]:
            band(ax, steps, reports, key, label, 'p05', 'p95', color)
    else:
        ax.text(.5,.5,'Supervised baseline: no safety critic',transform=ax.transAxes,ha='center')
    ax.set(title='Critic predictions: median and 5–95% range', ylabel='Predicted discounted margin (m)')
    fig.text(.07, .012, 'Offline diagnostics only. Shading shows state-distribution percentiles, not confidence intervals.', fontsize=10)
    save(fig, axes, output / 'checkpoint_comparison.png', 'Checkpoint comparison',
         f"Split: {reports[0]['split']} • {len(reports)} checkpoints • {reports[0]['actor']['rows']:,} evaluated states")
    with (output / 'checkpoint_comparison.csv').open('w', newline='') as file:
        fields = ['step', 'split', 'rows', *[k for k, _ in keys], 'median_longitudinal', 'median_steering']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for r in reports:
            a = r['actor']
            writer.writerow(dict(step=r['step'], split=r['split'], rows=a['rows'],
                                 **{k: a[k] for k, _ in keys},
                                 median_longitudinal=a['longitudinal']['median'], median_steering=a['steering']['median']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='Defaults to RUN_DIR/plots')
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    args = parser.parse_args()
    metrics = json.loads((args.run_dir / 'metrics.json').read_text())
    output = args.output or args.run_dir / 'plots'
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 10})
    plot_history(metrics, output)
    paths = [p for p in args.run_dir.glob('*/evaluation.json')
             if json.loads(p.read_text()).get('split') == args.split]
    if paths:
        plot_evaluations(metrics, paths, output)
    else:
        print(f'No {args.split} evaluation reports found; generated training history only.')


if __name__ == '__main__':
    main()
