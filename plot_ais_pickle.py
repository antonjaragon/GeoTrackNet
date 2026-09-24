#!/usr/bin/env python3
"""Quick offline inspection of csv2pkl_2021 output (trusted pickle files only).

Install: python3 -m pip install numpy matplotlib
Run: python3 plot_ais_pickle.py path/to/output_folder
     python3 plot_ais_pickle.py path/to/train_tracks.pkl --mmsi 338393067
     python3 plot_ais_pickle.py path/to/output_folder --no-show

Accepts a pickle or folder containing *_tracks.pkl. Creates one PNG per input
in INPUT_FOLDER/quicklook (override with --output-dir). Also shows plot windows
unless --no-show is supplied. Metadata ROI is used when present; otherwise
uses the fixed ROI from csv2pkl_2021. No map downloads or internet needed.

Expected dictionary: MMSI -> numeric array with columns
lat, lon, sog, cog, heading, rot, status, Unix UTC seconds, mmsi.
Plots a reproducible random sample of up to 100,000 messages. Statistics scan
ALL selected vessels, in chunks. Each pickle must still fit in RAM; files are
loaded sequentially. Missing optional fields (NaN) are expected. These checks
assess structure and ranges, not physical plausibility or model readiness.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import time

import numpy as np

DEFAULT_ROI = dict(south=24.1055773, north=26.7286216, west=-81.2981022, east=-77.262825)
COLS = ['latitude', 'longitude', 'sog', 'cog', 'heading', 'rot', 'status', 'timestamp', 'mmsi']


def utc(value):
    try:
        return datetime.fromtimestamp(value, timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    except (ValueError, OverflowError, OSError):
        return 'unavailable'


def inspect(path, args, plt):
    print(f'Loading {path.name} ({path.stat().st_size/2**20:.1f} MiB)...', flush=True)
    with path.open('rb') as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f'{path.name}: expected dictionary keyed by MMSI')
    roi = DEFAULT_ROI.copy()
    metadata = path.parent / 'metadata.json'
    if metadata.exists():
        roi.update(json.loads(metadata.read_text()).get('roi', {}))
    if not all(np.isfinite(roi[k]) for k in DEFAULT_ROI) or not (roi['south'] < roi['north'] and roi['west'] < roi['east']):
        raise ValueError('Invalid ROI in metadata.json')
    if args.mmsi is not None:
        data = {k: v for k, v in data.items() if str(k) == str(args.mmsi)}
        if not data:
            raise ValueError(f'MMSI {args.mmsi} not found in {path.name}')
    lengths, keys = [], []
    for key, a in data.items():
        if not isinstance(a, np.ndarray) or a.ndim != 2 or a.shape[1] != 9 or not np.issubdtype(a.dtype, np.number) or np.iscomplexobj(a):
            raise ValueError(f'{path.name}, MMSI {key}: expected real numeric (N,9) array')
        keys.append(key)
        lengths.append(len(a))
    total = sum(lengths)
    rng = np.random.default_rng(args.seed)
    indices = np.sort(rng.choice(total, size=min(total, args.max_points), replace=False)) if total else np.array([], dtype=int)
    sample = np.empty((len(indices), 9))
    missing = np.zeros(9, dtype=np.int64)
    checks, days = Counter(), Counter()
    speed_edges = np.linspace(0, 30, 61)
    speed_counts = np.zeros(60, dtype=np.int64)
    lower, upper = float('inf'), float('-inf')
    offset = 0
    last_report = time.monotonic()
    for key, size in zip(keys, lengths):
        a = data[key]
        left, right = np.searchsorted(indices, [offset, offset+size])
        sample[left:right] = a[indices[left:right]-offset]
        previous = None
        unsorted = False
        for start in range(0, size, 100000):
            b = a[start:start+100000]
            finite = np.isfinite(b)
            missing += (~finite).sum(axis=0)
            lat, lon, sog, cog, ts = b[:,0], b[:,1], b[:,2], b[:,3], b[:,7]
            valid_pos = finite[:,0] & finite[:,1]
            inside = (lat >= roi['south']) & (lat <= roi['north']) & (lon >= roi['west']) & (lon <= roi['east'])
            checks['outside_roi'] += int((valid_pos & ~inside).sum())
            checks['invalid_sog'] += int((finite[:,2] & ((sog < 0) | (sog > 30))).sum())
            checks['invalid_cog'] += int((finite[:,3] & ((cog < 0) | (cog >= 360))).sum())
            checks['mmsi_mismatch'] += int((b[:,8] != float(key)).sum())
            times = ts[finite[:,7]]
            if len(times):
                unsorted |= bool(np.any(np.diff(times) < 0) or (previous is not None and times[0] < previous))
                previous = times[-1]
                lower, upper = min(lower, times.min()), max(upper, times.max())
                # Only 2021 timestamps enter the daily panel; out-of-year values are counted.
                in_year = (times >= 1609459200) & (times < 1640995200)
                checks['outside_2021'] += int((~in_year).sum())
                dates, counts = np.unique((times[in_year] // 86400).astype(np.int64), return_counts=True)
                days.update({int(d): int(c) for d, c in zip(dates, counts)})
            speed_counts += np.histogram(sog[finite[:,2]], bins=speed_edges)[0]
            if time.monotonic()-last_report >= 5:
                print(f'  Checked {offset+start+len(b):,}/{total:,} messages', flush=True)
                last_report = time.monotonic()
        checks['unsorted_vessels'] += int(unsorted)
        offset += size
    del data
    print(f'\n{path.name}: {len(keys):,} vessels, {total:,} messages')
    print(f'UTC range: {utc(lower)} — {utc(upper)}')
    for name, value in checks.items():
        print(f'  {name}: {value:,}')
    print('  Nonfinite values: ' + ', '.join(f'{k}={n:,}' for k,n in zip(COLS, missing)))
    print('  Missing heading/ROT/status can be expected; required fields should be finite.', flush=True)

    fig, axs = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    fig.suptitle(f'{path.name}  |  {len(keys):,} vessels · {total:,} messages', fontsize=15)
    ax = axs[0,0]
    good = np.isfinite(sample[:,0]) & np.isfinite(sample[:,1])
    points = sample[good]
    if len(points):
        colored = np.isfinite(points[:,2])
        scatter = ax.scatter(points[colored,1], points[colored,0], c=points[colored,2], s=2,
                             cmap='viridis', vmin=0, vmax=30, alpha=.65, rasterized=True)
        ax.scatter(points[~colored,1], points[~colored,0], color='grey', s=3)
        fig.colorbar(scatter, ax=ax, label='SOG (knots)', shrink=.8)
    ax.plot([roi['west'],roi['east'],roi['east'],roi['west'],roi['west']],
            [roi['south'],roi['south'],roi['north'],roi['north'],roi['south']],
            '--', color='crimson', linewidth=1.2, label='ROI')
    # Include all sampled positions so out-of-ROI points stay visible.
    ax.update_datalim([[roi['west'],roi['south']], [roi['east'],roi['north']]])
    ax.autoscale_view()
    ax.set_aspect(1/np.cos(np.deg2rad((roi['south']+roi['north'])/2)))
    ax.set(xlabel='Longitude (°)', ylabel='Latitude (°)', title=f'Positions: {len(points):,} sampled points (no basemap)')
    ax.legend(fontsize=8)
    axs[0,1].stairs(speed_counts, speed_edges, fill=True, color='#357b95')
    axs[0,1].set(xlabel='SOG (knots)', ylabel='Messages', title='Speed distribution · all selected messages')
    if days:
        first, last = min(days), max(days)
        dates = np.arange(first,last+1)
        axs[1,0].plot(dates.astype('datetime64[D]'), [days.get(int(d),0) for d in dates], color='#357b95')
        axs[1,0].tick_params(axis='x', rotation=25)
    axs[1,0].set(xlabel='Date (UTC)', ylabel='Messages per day', title='2021 coverage · all selected messages')
    if lengths:
        axs[1,1].hist(lengths, bins=min(40,max(1,len(lengths))), color='#357b95')
    axs[1,1].set(xlabel='Messages per vessel', ylabel='Vessels', title='Track sizes · all selected vessels')
    for axis in axs.flat:
        axis.grid(alpha=.2)
    output = args.output_dir or path.parent / 'quicklook'
    output.mkdir(parents=True, exist_ok=True)
    suffix = f'_mmsi_{args.mmsi}' if args.mmsi is not None else ''
    destination = output / f'{path.stem}{suffix}_quicklook.png'
    fig.savefig(destination, dpi=150)
    print(f'Saved {destination}', flush=True)
    if not args.no_show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', type=Path, help='Pickle file or converter output folder')
    parser.add_argument('--mmsi', type=int, help='Inspect only one vessel')
    parser.add_argument('--max-points', type=int, default=100000, help='Maximum plotted positions (default 100000)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--no-show', action='store_true', help='Save PNGs without opening windows')
    args = parser.parse_args()
    if args.max_points < 1 or args.seed < 0:
        parser.error('--max-points must be positive and --seed nonnegative')
    files = sorted(args.input.glob('*_tracks.pkl')) if args.input.is_dir() else [args.input]
    if not files or any(not p.is_file() for p in files):
        parser.error('No matching pickle files found')
    import matplotlib
    if args.no_show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    failed = 0
    for path in files:
        try:
            inspect(path, args, plt)
        except Exception as exc:
            failed += 1
            plt.close('all')
            print(f'ERROR {path.name}: {exc}', flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
