"""Refreshable, offline plots of completed online HJ evaluations.

Also usable after training: python -m training.plot_online_hj_evaluation RUN_DIR
"""
import argparse
from datetime import datetime, timezone
from html import escape
import io
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.ticker import PercentFormatter


COLORS = {'pid':'#64748b', 'safety':'#d97706', 'filter':'#0284c7'}
LABELS = {'pid':'PID', 'safety':'Safety actor', 'filter':'PID + safety filter'}


def evaluation_figure(records, title='All matched scenarios', nominal_kind='pid'):
    labels = LABELS if nominal_kind == 'pid' else dict(pid='Goal actor',safety='Safety actor',filter='SAGE (goal + safety)')
    fig = Figure(figsize=(13, 10), layout='constrained', facecolor='#f8fafc')
    FigureCanvasAgg(fig)
    axes = fig.subplots(3, 2).ravel()
    steps = [r['step'] for r in records]
    def values(mode, key):
        return [np.nan if r['modes'][mode].get(key) is None else r['modes'][mode][key] for r in records]
    for index, key, title_text, ylabel in [
        (0,'collision','Observed collisions ↓','Fraction of matched episodes'),
        (2,'intervention_fraction','Filter interventions','Fraction of decisions'),
        (3,'mean_speed','Mean ego speed','m/s'),
        (4,'mean_absolute_gap_error','Following gap error ↓','Mean absolute error (m)')]:
        ax = axes[index]
        modes = ('filter',) if index == 2 else COLORS
        for mode in modes:
            ax.plot(steps,values(mode,key),marker='o',markersize=4,label=labels[mode],color=COLORS[mode])
        ax.set(title=title_text,ylabel=ylabel); ax.legend(fontsize=8)
    ax = axes[1]
    for key,label,color in [('collision_rate_when_predicted_safe','Collision rate among safe predictions','#dc2626'),
                            ('missed_collision_fraction','Fraction of collisions predicted safe','#9333ea')]:
        ax.plot(steps,values('pid_then_safety',key),marker='o',markersize=4,label=label,color=color)
    ax.set(title='Value warnings on recovery probes ↓',ylabel='Fraction of eligible probes')
    ax.legend(fontsize=8)
    ax = axes[5]
    for key,label,color in [('paired_cases','Matched starts','#059669'),('excluded_cases','Excluded starts','#dc2626')]:
        ax.plot(steps,[r[key] for r in records],marker='o',markersize=4,label=label,color=color)
    ax.plot(steps,values('pid_then_safety','predicted_safe_cases'),marker='o',markersize=4,
            label='Predicted-safe probes',color='#9333ea',linestyle='--')
    ax.set(title='Evaluation sample counts',ylabel='Count'); ax.legend(fontsize=8)
    for index,ax in enumerate(axes):
        ax.set_xlabel('Training environment steps'); ax.grid(alpha=.2)
        ax.set_facecolor('white'); ax.ticklabel_format(axis='x',style='plain')
        if index in (0,1,2):
            ax.set_ylim(-.03,1.03); ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        else: ax.set_ylim(bottom=0)
    fig.suptitle(f'Online HJ evaluation — {title}\nLatest training step: {steps[-1]:,}',fontsize=15)
    return fig


def plot_evaluations(outdir):
    """Atomically update PNG and a self-refreshing HTML dashboard. No server needed."""
    outdir = Path(outdir)
    records = [json.loads(line) for line in (outdir/'evaluation.jsonl').read_text().splitlines() if line.strip()]
    if not records: return
    nominal_kind=records[-1].get('nominal_kind','pid')
    # Summary lines are complete before this function is invoked by the trainer.
    scopes = [('All matched scenarios',records)]
    for profile in records[-1].get('by_profile',{}):
        scopes.append((f'{"Policy" if nominal_kind == "goal" else "PID profile"}: {profile}',[dict(step=r['step'],**r['by_profile'][profile]) for r in records]))
    for label in records[-1].get('by_start',{}):
        scopes.append((f'{label.capitalize()} starts',[dict(step=r['step'],**r['by_start'][label]) for r in records]))
    for label,title in [('survived_horizon','Full braking survived the horizon'),
                        ('collision','Full braking collided (not proven unavoidable)'),
                        ('incomplete','Full braking ended before the horizon')]:
        if label in records[-1].get('by_braking',{}) and any(
                r['by_braking'][label]['paired_cases']+r['by_braking'][label]['excluded_cases'] for r in records):
            scopes.append((title,[dict(step=r['step'],**r['by_braking'][label]) for r in records]))
    panels = []
    for index,(title,history) in enumerate(scopes):
        fig = evaluation_figure(history,title,nominal_kind)
        try:
            if index == 0:
                temporary = outdir/'evaluation_plot.tmp'
                fig.savefig(temporary,format='png',dpi=130)
                temporary.replace(outdir/'evaluation_plot.png')
            svg = io.StringIO(); fig.savefig(svg,format='svg')
            # Strip the XML prolog/doctype before embedding in HTML.
            content = svg.getvalue(); content = content[content.index('<svg'):]
            panels.append(f'<section id="scope-{index}" {"hidden" if index else ""}>{content}</section>')
        finally: fig.clear()
    options = ''.join(f'<option value="scope-{i}">{escape(title)}</option>' for i,(title,_) in enumerate(scopes))
    updated = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    proposer='goal-policy' if nominal_kind == 'goal' else 'PID'
    baseline=records[-1].get('braking_baseline')
    baseline_text=(f'Immediate full-braking reference: {baseline["collisions"]}/{baseline["cases"]} collisions; '
                   f'{baseline["survived_horizon"]}/{baseline["cases"]} survived the full horizon. '
                   'Use the selector to inspect these groups separately.' if baseline else '')
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="10"><title>Online HJ evaluation</title>
<style>body{{font:16px system-ui;margin:20px;background:#f8fafc;color:#0f172a}}
main{{max-width:1300px;margin:auto}}select{{font:inherit;padding:6px}}svg{{width:100%;height:auto}}
p{{max-width:1100px;line-height:1.5}}.muted{{color:#475569}}[hidden]{{display:none}}</style></head>
<body><main><h1>Online HJ evaluation</h1><p class="muted">Updated {updated} · Training step {records[-1]['step']:,}.
This page reloads every 10 seconds; charts update after each completed evaluation.</p>
<label>Show <select id="scope">{options}</select></label>
<p>Lower collision and false-safe rates are better. Read speed and gap error alongside safety:
stopping indefinitely can reduce collisions. Intervention rate alone is not a quality score.
Missing measurements appear as gaps, not zeros.</p>
<p>{baseline_text}</p>
{''.join(panels)}
<p class="muted">Small, finite-horizon screening suite; collision-free rollouts are not safety certificates.
Recovery probes test a {proposer} action after collecting real frame history under the frozen safety actor,
then continue with that actor. No optimal HJ ground truth is available.
Changing matched-case counts can affect comparisons; check the count panel and rollout logs.</p>
</main><script>const select=document.getElementById('scope');
const chosen=location.hash.slice(1);if([...select.options].some(o=>o.value===chosen))select.value=chosen;
function show(){{document.querySelectorAll('section').forEach(s=>s.hidden=s.id!==select.value);}}
select.onchange=()=>{{location.hash=select.value;show();}};show();</script></body></html>'''
    temporary = outdir/'evaluation_dashboard.tmp'
    temporary.write_text(html,encoding='utf-8'); temporary.replace(outdir/'evaluation_dashboard.html')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('outdir',type=Path)
    plot_evaluations(parser.parse_args().outdir)
