"""Plot saved CARLA traces and export per-step CSV; no simulator required."""
import argparse
import csv
import itertools
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'_local'/'mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_trace(trace,summary,dt,output):
    rows=trace['rows']
    if not rows:
        return
    prefix=f"{summary['checkpoint']}_{summary['scenario_id']}_{summary['mode']}"
    t=[(row['frame']-trace['initial_frame'])*dt for row in rows]
    gap_t=[0.,*[(row['next_frame']-trace['initial_frame'])*dt for row in rows]]
    gaps=[trace['initial_gap_m'],*[row['next_gap_m'] for row in rows]]
    low,high=trace['d_min'],trace['d_max']
    fig,axes=plt.subplots(3,2,figsize=(13,10))
    fig.suptitle(f"{summary['checkpoint']} / {summary['scenario_id']} / {summary['mode']}",fontsize=16)
    ax=axes[0,0]
    ax.axhspan(low,high,color='#31956a',alpha=.1,label='Permitted gap')
    ax.plot(gap_t,gaps,color='#2676b8',label='Measured gap')
    for bound in (low,high):ax.axhline(bound,color='#bd4a38',linestyle='--')
    ax.fill_between(gap_t,gaps,high,where=[g>=high for g in gaps],color='#bd4a38',alpha=.25)
    ax.set(title='Following gap',ylabel='Gap (m)',ylim=(min(0.,min(gaps)),max(high+1,max(gaps)+.5)))
    ax=axes[0,1]
    ax.plot(t,[r['closing_speed_mps'] if r['closing_speed_mps'] is not None else float('nan') for r in rows])
    ax.axhline(0,color='gray',linewidth=.8)
    ax.set(title='Closing speed (positive = approaching leader)',ylabel='Closing speed (m/s)')
    for ax,col,title in ((axes[1,0],0,'Longitudinal request'),(axes[1,1],1,'Steering request')):
        ax.plot(t,[r['executed_request'][col] for r in rows],label='Executed',color='#2676b8')
        ax.plot(t,[r['nominal_action'][col] for r in rows],label='Nominal PID proposal',color='#d66c28',linestyle='--',alpha=.8)
        ax.axhline(0,color='gray',linewidth=.8)
        ax.set(title=title,ylabel='Requested command',ylim=((-3.1,3.1) if col==0 else (-1.05,1.05)))
    ax=axes[2,0]
    for field,label in [('ego_travel_m','Ego'),('leader_travel_m','Leader')]:
        ax.plot(gap_t,[0.,*itertools.accumulate(r[field] for r in rows)],label=label)
    ax.set(title='Cumulative path length',ylabel='Distance travelled (m)')
    ax=axes[2,1];ax.axis('off')
    lines=[f"Steps: {summary['steps']} ({len(rows)*dt:.1f} simulation seconds)",
           f"Any collision: {summary['any_collision']}",f"Lane departure onsets: {summary['lane_departures']}",
           f"Gap range: {min(gaps):.2f}–{max(gaps):.2f} m",
           f"Gap samples within bounds: {100*summary['permitted_gap_fraction']:.1f}%",
           f"Upper-bound violation: {summary['far_violation']}",
           f"Interventions: {summary['interventions'] if summary['interventions'] is not None else 'not applicable'}",
           '', 'First action uses nominal PID for history warmup.',
           'Commands are requests, not measured acceleration.',
           'Camera images/video were not saved by this run.']
    ax.text(.02,.95,'\n'.join(lines),va='top',fontsize=11,transform=ax.transAxes)
    for ax in axes.flat:
        if ax is axes[2,1]:continue
        ax.set_xlabel('Simulation time (s)');ax.grid(alpha=.2)
        if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=9)
    fig.tight_layout(rect=(0,0,1,.95))
    path=output/f'{prefix}.png';fig.savefig(path,dpi=150);plt.close(fig)
    with (output/f'{prefix}_steps.csv').open('w',newline='') as f:
        fields=['time_s','frame','gap_m','next_gap_m','closing_speed_mps','longitudinal_request',
                'steering_request','nominal_longitudinal','nominal_steering','intervened']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for time_s,row in zip(t,rows):
            writer.writerow(dict(time_s=time_s,frame=row['frame'],gap_m=row['gap_m'],next_gap_m=row['next_gap_m'],
                                 closing_speed_mps=row['closing_speed_mps'],longitudinal_request=row['executed_request'][0],
                                 steering_request=row['executed_request'][1],nominal_longitudinal=row['nominal_action'][0],
                                 nominal_steering=row['nominal_action'][1],intervened=row['intervened']))
    print(path)


def plot_comparison(results, dt, output):
    labels = {
        'actor_only': ('Actor only', '#2676b8', '-'),
        'safety_filter': ('Safety filter', '#d66c28', '--'),
        'pid': ('Nominal PID', '#31956a', '-.'),
    }
    runs = [results[mode] for mode in labels if mode in results and results[mode]['trace']['rows']]
    if len(runs) < 2:
        return

    summary = runs[0]['summary']
    trace = runs[0]['trace']
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    gap_ax, speed_ax, longitudinal_ax, steering_ax = axes.flat
    gap_ax.axhspan(trace['d_min'], trace['d_max'], color='gray', alpha=.12, label='Permitted gap')
    for result in runs:
        trace = result['trace']
        rows = trace['rows']
        label, color, linestyle = labels[result['summary']['mode']]
        style = dict(label=label, color=color, linestyle=linestyle)
        times = [(row['frame'] - trace['initial_frame']) * dt for row in rows]
        gap_times = [0., *((row['next_frame'] - trace['initial_frame']) * dt for row in rows)]
        gap_ax.plot(gap_times, [trace['initial_gap_m'], *(row['next_gap_m'] for row in rows)], **style)
        speed_ax.plot(times, [row['closing_speed_mps'] if row['closing_speed_mps'] is not None
                              else float('nan') for row in rows], **style)
        longitudinal_ax.plot(times, [row['executed_request'][0] for row in rows], **style)
        steering_ax.plot(times, [row['executed_request'][1] for row in rows], **style)

    gap_ax.set(title='Following gap', ylabel='Gap (m)')
    speed_ax.set(title='Closing speed (positive = approaching leader)', ylabel='Closing speed (m/s)')
    longitudinal_ax.set(title='Executed longitudinal request', ylabel='Requested command', ylim=(-3.1, 3.1))
    steering_ax.set(title='Executed steering request', ylabel='Requested command', ylim=(-1.05, 1.05))
    for ax in axes.flat:
        ax.grid(alpha=.2)
    for ax in axes[1]:
        ax.set_xlabel('Simulation time (s)')
    gap_ax.legend()
    fig.suptitle(f"{summary['checkpoint']} / {summary['scenario_id']} / Mode comparison")
    fig.tight_layout()
    path = output / f"{summary['checkpoint']}_{summary['scenario_id']}_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    manifest=json.loads((args.results/'manifest.json').read_text())
    output=args.output or args.results/'plots';output.mkdir(parents=True,exist_ok=True)
    files=sorted(args.results.glob('checkpoint_*/scenario_*.json'))
    if not files:parser.error('No saved scenario traces found')
    for path in files:
        results = json.loads(path.read_text())
        for result in results.values():
            plot_trace(result['trace'],result['summary'],manifest['dt'],output)
        plot_comparison(results, manifest['dt'], output)


if __name__=='__main__':main()
