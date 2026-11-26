"""Plot completed braking runs in the paired selection experiment."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('training/_local/conformal_selection_90_s20261124/complete.json'))
    parser.add_argument('--output', type=Path, default=Path('training/_local/conformal_selection_90_s20261124/outcomes_by_environment'))
    args = parser.parse_args()
    data = json.loads(args.results.read_text())['by_condition']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'svg.fonttype': 'none'})
    conditions = [f'{t}_{w}' for t in ('day', 'sunset', 'night') for w in ('sunny', 'cloudy', 'rainy')]
    fig, ax = plt.subplots(figsize=(14, 4.8))
    width = .28
    for i, condition in enumerate(conditions):
        item = data[condition]
        for x, mode, face, text_color in ((i-width/2, 'raw_min', 'white', 'black'),
                                           (i+width/2, 'lower_min', 'black', 'white')):
            value = item[mode]['braking_complete']
            ax.bar(x, value, width=width, facecolor=face, edgecolor='black', linewidth=1.1, zorder=3)
            if value:
                ax.text(x, value-.32, str(value), ha='center', va='top', color=text_color,
                        fontsize=10, fontweight='bold', zorder=4)
    ax.set_xticks(range(9), [name.replace('_', '\n').title() for name in conditions])
    ax.set_ylim(0, 10.8)
    ax.set_yticks(range(0, 11, 2))
    ax.set_ylabel('Completed braking trajectories')
    ax.set_title('Controller-selection outcomes by environment', loc='left', fontsize=14, fontweight='bold', pad=12)
    ax.grid(axis='y', color='#d8d8d8', linewidth=.7, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='x', length=0, pad=8)
    ax.legend(handles=[Patch(facecolor='white', edgecolor='black', label='Raw minimum'),
                       Patch(facecolor='black', edgecolor='black', label='Conformal lower-bound minimum')],
              frameon=False, loc='upper right', ncol=2)
    fig.text(.075, .025, '10 matched scenarios per environment', fontsize=9, color='#505050')
    fig.tight_layout(rect=(0, .06, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(args.output.with_suffix('.' + ext), dpi=300, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
