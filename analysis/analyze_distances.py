"""Audit stored post-action distances and write dependency-free SVG histograms.

Run from repository root: .venv/bin/python analysis/analyze_distances.py
Only numeric trajectory metadata is read; source datasets are never modified.
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch

OUT = Path('analysis/distance_distributions')
ROOTS = ['carla_following', 'full_range_300']
CONDITIONS = [f'{t}_{w}' for t in ['day', 'sunset', 'night']
              for w in ['sunny', 'cloudy', 'rainy']]
CUTOFFS = [8, 10, 12, 15, 20, 25, 30]


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def chart(data, path, title, conditions):
    # Common 1 m bins, full data range, and shared y scale across panels.
    xmax = max(1, int(np.ceil(max(a.max() for a in data.values()))))
    bins = np.arange(0, xmax + 1, 1)
    hist = {k: np.histogram(a, bins=bins)[0] / len(a) * 100
            for k, a in data.items()}
    ymax = max(v.max() for v in hist.values()) * 1.1
    width, height = 1380, 90 + 300 * len(conditions)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<style>text{font-family:Arial,sans-serif;fill:#172b42;font-size:13px}</style>',
             f'<text x="30" y="30" style="font-size:23px">{title}</text>',
             '<text x="30" y="56">Stored unsigned edge distances • 1 m bins • percentage of recorded steps • all values included</text>']
    for row, condition in enumerate(conditions):
        for col, root in enumerate(ROOTS):
            key = (root, condition)
            a, values = data[key], hist[key]
            left, top, w, h = 70 + col * 680, 120 + row * 300, 590, 190
            parts.append(f'<text x="{left}" y="{top-20}" style="font-size:17px">{root} / {condition} (n={len(a):,})</text>')
            for tick in np.linspace(0, ymax, 5):
                y = top + h - tick / ymax * h
                parts.extend([f'<path d="M{left} {y}h{w}" stroke="#e1e6eb"/>',
                              f'<text x="{left-8}" y="{y+4}" text-anchor="end">{tick:.0f}%</text>'])
            for i, v in enumerate(values):
                bh = v / ymax * h
                parts.append(f'<rect x="{left+i/xmax*w:.2f}" y="{top+h-bh:.2f}" width="{w/xmax:.2f}" height="{bh:.2f}" fill="{["#2976b8", "#dc8036"][col]}"/>')
            for tick in range(0, xmax+1, 5):
                x = left + tick / xmax * w
                parts.append(f'<text x="{x}" y="{top+h+20}" text-anchor="middle">{tick}</text>')
            parts.append(f'<text x="{left+w/2}" y="{top+h+43}" text-anchor="middle">Stored distance (m)</text>')
    parts.append('</svg>')
    path.write_text('\n'.join(parts))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data, rows, audit = {}, [], []
    for root in ROOTS:
        for condition in CONDITIONS:
            folder = Path('dataset') / root / condition
            ds, costs, lengths, states = [load(folder / f'{name}.pth')
                                        for name in ['distances', 'costs', 'seq_lengths', 'states']]
            assert len(ds) == len(costs) == len(lengths) == len(states)
            for d, c, n, s in zip(ds, costs, lengths, states):
                assert d.ndim == 1 and len(d) == len(c) == int(n) == len(s)
                assert torch.isfinite(d).all() and (d >= 0).all()
            a = np.concatenate([d.numpy() for d in ds])
            c = np.concatenate([v.numpy() for v in costs]) > 0
            data[root, condition] = a
            row = dict(dataset=root, condition=condition, episodes=len(ds), steps=len(a),
                       minimum=float(a.min()), maximum=float(a.max()),
                       **{f'p{q}': float(np.percentile(a, q)) for q in [5, 25, 50, 75, 90, 95, 99]},
                       pct_below_1=float(100*np.mean(a < 1)), collision_steps=int(c.sum()),
                       collision_steps_distance_ge_1=int(np.sum(c & (a >= 1))))
            for cutoff in CUTOFFS:
                row[f'pct_steps_gt_{cutoff}'] = float(100*np.mean(a > cutoff))
                row[f'pct_episodes_any_gt_{cutoff}'] = float(100*np.mean([bool((d > cutoff).any()) for d in ds]))
            rows.append(row)
            # Compare d[t] to centers from s[t+1], which correspond to the same time.
            centers, gaps = [], []
            for d, s in zip(ds, states):
                centers.extend(torch.linalg.vector_norm(s[1:, :3]-s[1:, 6:9], dim=1).tolist())
                gaps.extend(d[:-1].tolist())
            difference = np.asarray(centers)-np.asarray(gaps)
            audit.append(dict(dataset=root, condition=condition,
                              aligned_center_minus_stored_quantiles=np.percentile(difference, [0, 5, 50, 95, 100]).tolist()))
        data[root, 'pooled'] = np.concatenate([data[root, c] for c in CONDITIONS])
    with (OUT/'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (OUT/'audit.json').write_text(json.dumps(audit, indent=2))
    chart(data, OUT/'by_condition.svg', 'Following distance distributions by condition', CONDITIONS)
    chart(data, OUT/'pooled.svg', 'Following distance distributions pooled over nine conditions', ['pooled'])
    for root in ROOTS:
        a = data[root, 'pooled']
        print(root, 'steps', len(a), 'quantiles', np.percentile(a, [0, 50, 90, 95, 99, 100]))
        for cutoff in CUTOFFS:
            print(' >', cutoff, 'm:', round(100*np.mean(a>cutoff), 3), '%; per-condition range:',
                  [round(f([r[f'pct_steps_gt_{cutoff}'] for r in rows if r['dataset']==root]), 3) for f in [min, max]])
    print('Audit:', [(r['dataset'], r['condition'], r['episodes'], r['collision_steps'], r['collision_steps_distance_ge_1']) for r in rows])


if __name__ == '__main__':
    main()
