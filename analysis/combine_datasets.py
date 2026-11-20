"""Combine the two following collections without modifying their source files.

Run from repository root: .venv/bin/python analysis/combine_datasets.py
Images use independent reflink copies when supported, otherwise regular copies.
"""
import json
import os
from pathlib import Path
import subprocess

import torch

ROOT = Path('dataset')
OUTPUT = ROOT / 'combined_800'
STAGING = ROOT / '.combined_800_incomplete'
SOURCES = [('carla_following', 500), ('full_range_300', 300)]
FIELDS = {'actions': (2,), 'states': (12,), 'costs': (), 'distances': ()}
CONDITIONS = [f'{time}_{weather}' for time in ('day', 'sunset', 'night')
              for weather in ('sunny', 'cloudy', 'rainy')]


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def main():
    if OUTPUT.exists() or STAGING.exists():
        raise FileExistsError('Output or staging folder already exists; refusing to overwrite.')
    # Validate every numerical trajectory before writing the combined collection.
    bundles = {}
    for condition in CONDITIONS:
        combined = {field: [] for field in FIELDS}
        all_lengths = []
        for source, count in SOURCES:
            folder = ROOT / source / condition
            lengths = load(folder / 'seq_lengths.pth')
            assert lengths.dtype == torch.int64 and lengths.shape == (count,)
            assert bool((lengths > 0).all())
            images = sorted((folder / 'obses').glob('*.pth'))
            assert [p.name for p in images] == [f'episode_{i:03d}.pth' for i in range(count)]
            for field, trailing in FIELDS.items():
                values = load(folder / f'{field}.pth')
                assert isinstance(values, list) and len(values) == count
                for value, n in zip(values, lengths.tolist()):
                    assert value.dtype == torch.float32 and value.shape == (n, *trailing)
                    assert bool(torch.isfinite(value).all())
                combined[field].extend(values)
            all_lengths.append(lengths)
        bundles[condition] = (combined, torch.cat(all_lengths))
    STAGING.mkdir()
    summary = {}
    for condition, (combined, lengths) in bundles.items():
        folder = STAGING / condition
        (folder / 'obses').mkdir(parents=True)
        for field, values in combined.items():
            dest = folder / f'{field}.pth'
            torch.save(values, dest)
            restored = load(dest)
            assert len(restored) == 800
            assert all(torch.equal(a, b) for a, b in zip(values, restored))
        torch.save(lengths, folder / 'seq_lengths.pth')
        assert torch.equal(load(folder / 'seq_lengths.pth'), lengths)
        offset = 0
        for source, count in SOURCES:
            for i in range(count):
                src = ROOT / source / condition / 'obses' / f'episode_{i:03d}.pth'
                dst = folder / 'obses' / f'episode_{offset+i:03d}.pth'
                subprocess.run(['cp', '--reflink=auto', '--', str(src), str(dst)], check=True)
                assert src.stat().st_size == dst.stat().st_size
                assert (src.stat().st_dev, src.stat().st_ino) != (dst.stat().st_dev, dst.stat().st_ino)
            # Check first and last image trajectories in each imported block.
            for i in (0, count-1):
                src = ROOT / source / condition / 'obses' / f'episode_{i:03d}.pth'
                target = offset + i
                image = load(folder / 'obses' / f'episode_{target:03d}.pth')
                assert image.dtype == torch.uint8
                assert image.shape == (int(lengths[target]), 224, 224, 3)
                assert torch.equal(image, load(src))
            offset += count
        assert len(list((folder / 'obses').glob('*.pth'))) == 800
        summary[condition] = {'episodes': 800, 'steps': int(lengths.sum())}
        print(f'{condition}: verified 800 episodes, {int(lengths.sum()):,} steps', flush=True)
    manifest = {'sources': [{'folder': name, 'count_per_condition': count} for name, count in SOURCES],
                'mapping': {'episode_000.pth..episode_499.pth': 'carla_following episodes 000..499',
                            'episode_500.pth..episode_799.pth': 'full_range_300 episodes 000..299'},
                'format': 'unchanged: float32 tensor lists, int64 lengths, uint8 RGB image tensors',
                'verification': 'All numeric tensors round-trip exactly; all image names, counts, sizes and independent inodes checked; first/last images per source per condition compared exactly.',
                'conditions': summary}
    (STAGING / 'merge_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    os.rename(STAGING, OUTPUT)
    print(f'Complete: {OUTPUT} (7,200 trajectories)', flush=True)


if __name__ == '__main__':
    main()
