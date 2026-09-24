#!/usr/bin/env python3
"""Convert 2021 AIS CSV/CSV.GZ files to chronological vessel-track pickles.

Install: python -m pip install numpy psutil
Monitoring: --progress-seconds 5; console + run.log, per-file/stage stats in metadata.json.
psutil is optional; without it, RAM metrics are unavailable. Reading ETA is approximate.
Run: python csv2pkl.py --input-dir ./ais_2021_csv --workers 2 --train 80 --valid 10 --test 10

Generate missing mean without rereading CSVs:
  python csv2pkl.py --mean-only path/to/train_tracks.pkl
For already normalized, preprocessed training tracks, add --mean-input-normalized.
Only load pickle files you trust. Each training pickle must fit in RAM.
mean.pkl is a float64 mean four-hot vector, computed from TRAINING messages only.
Default resolutions match flags_config.py: 0.01 deg latitude/longitude, 1 knot,
5 deg COG. With this ROI the bin counts are 263,404,30,72 (769 values).
GeoTrackNet must use this same ROI/resolutions. Raw track outputs remain raw;
this does not replace dataset_preprocessing.py. Recompute mean after any
resampling/filtering/normalization that changes the training dataset.
mean_metadata.json records the encoding used. Empty training sets are rejected.

Input is required via --input-dir for full conversion. Use --workers 2 (default) or --workers 4.
Workers parse independent files with bounded batches and temporary SQLite files.
Merging and export are sequential; one complete split must still fit in RAM.
Percentages refer to retained AIS messages, ordered globally by UTC timestamp.
Equal timestamps stay together, so achieved percentages can differ slightly.
The same vessel may occur in multiple splits, at different times. No voyage
segmentation, interpolation, normalization, or deduplication is performed.
Output: dict[int MMSI, float64 ndarray of shape (N, 9)], time sorted, columns:
latitude, longitude, sog, cog, heading, rot, navigation_status, timestamp, mmsi.
Missing optional fields are NaN; downstream code must handle these if used.
SQLite stages filtered data on disk; exporting needs RAM for one split at a time.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import queue
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import io
import threading
import shutil
import json
import math
from pathlib import Path
import pickle
import sqlite3
import tempfile
import time

import numpy as np

try:
    import psutil  # Optional: current RAM and system memory availability.
except ImportError:
    psutil = None


def duration(seconds):
    if seconds is None:
        return 'estimating'
    seconds = max(0, int(seconds))
    return f'{seconds // 3600:d}h {(seconds % 3600) // 60:02d}m {seconds % 60:02d}s'


class Monitor:
    """Periodic heartbeat, including while SQLite or pickle is busy.

    Reading ETA uses physical input bytes, including gzip read-ahead; it is
    approximate and excludes later indexing/export work. CPU is process CPU
    (100% = one fully used core). psutil is optional on macOS/Linux/Windows.
    """
    def __init__(self, output, counts, total_bytes, interval):
        self.output, self.counts = output, counts
        self.total_bytes, self.interval = total_bytes, interval
        self.started = self.phase_started = time.monotonic()
        self.phase = 'starting'
        self.phase_seconds = Counter()
        self.completed_bytes = 0
        self.raw = None
        self.file_size = 0
        self.file_label = ''
        self.done = self.total = 0
        self.last_rows = 0
        self.last_time = self.started
        self.last_cpu = time.process_time()
        self.peak_memory = 0
        self.process = psutil.Process() if psutil else None
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.logfile = (output / 'run.log').open('w', encoding='utf-8', buffering=1)
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def log(self, message, **_):
        with self.lock:
            line = f'[{datetime.now().astimezone():%H:%M:%S}] {message}'
            print(line, flush=True)
            self.logfile.write(line + '\n')

    def set_phase(self, phase, total=0):
        with self.lock:
            now = time.monotonic()
            self.phase_seconds[self.phase] += now - self.phase_started
            self.phase, self.phase_started = phase, now
            self.done, self.total = 0, total
        self.log(f'Stage: {phase}')

    def snapshot(self):
        with self.lock:
            now = time.monotonic()
            dt = max(now - self.last_time, 1e-6)
            rows = self.counts['rows_read']
            cpu = time.process_time()
            parts = [self.phase, f'elapsed {duration(now-self.started)}',
                     f'parent CPU {(cpu-self.last_cpu)/dt*100:.0f}%']
            if self.process:
                ram = self.process.memory_info().rss
                for child in self.process.children(recursive=True):
                    try:
                        ram += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                self.peak_memory = max(self.peak_memory, ram)
                available = psutil.virtual_memory().available
                parts += [f'RAM incl. workers {ram/2**30:.2f} GiB', f'available {available/2**30:.2f} GiB']
                if available < 512 * 2**20:
                    parts.append('LOW AVAILABLE MEMORY')
            else:
                parts.append('RAM unavailable (install psutil)')
            free = shutil.disk_usage(self.output).free
            parts.append(f'disk free {free/2**30:.1f} GiB')
            if free < 2**30:
                parts.append('LOW DISK SPACE')
            if self.phase == 'reading':
                position = 0
                if self.raw is not None:
                    try:
                        position = min(self.raw.tell(), self.file_size)
                    except (ValueError, OSError):
                        pass
                read = min(self.total_bytes, self.completed_bytes + position)
                elapsed = max(now-self.phase_started, 1e-6)
                eta = (self.total_bytes-read)/(read/elapsed) if read else None
                kept = self.counts['kept']
                parts += [self.file_label, f'input bytes {100*read/max(self.total_bytes,1):.1f}%',
                          f'{rows:,} rows', f'{(rows-self.last_rows)/dt:,.0f} rows/s',
                          f'kept {kept:,} ({100*kept/max(rows,1):.2f}%)',
                          f'reading ETA ~{duration(eta)}']
                if rows >= 10000 and kept == 0:
                    parts.append('No matching rows yet: check ROI/year/filter counts')
            elif self.total:
                elapsed = max(now-self.phase_started, 1e-6)
                eta = (self.total-self.done)/(self.done/elapsed) if self.done else None
                parts += [f'{self.done:,}/{self.total:,} ({100*self.done/self.total:.1f}%)',
                          f'stage ETA ~{duration(eta)}']
            else:
                parts.append(f'stage active for {duration(now-self.phase_started)}; ETA unavailable')
            self.last_time, self.last_rows, self.last_cpu = now, rows, cpu
            self.log(' | '.join(parts))

    def run(self):
        while not self.stop_event.wait(self.interval):
            try:
                self.snapshot()
            except Exception as exc:
                self.log(f'Monitor sample unavailable: {exc}')

    def close(self):
        self.stop_event.set()
        self.thread.join()
        with self.lock:
            self.phase_seconds[self.phase] += time.monotonic()-self.phase_started
            self.logfile.close()

# User's fixed region of interest (inclusive boundaries).
SOUTH = 24.1055773
NORTH = 26.7286216
WEST = -81.2981022
EAST = -77.262825
YEAR = 2021
SOG_MAX = 30.0
COLUMNS = ['latitude', 'longitude', 'sog', 'cog', 'heading', 'rot',
           'navigation_status', 'timestamp', 'mmsi']
REQUIRED = {'mmsi', 'base_date_time', 'longitude', 'latitude', 'sog', 'cog'}


def optional_number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def parse_row(row, cargo_only):
    lat, lon = float(row['latitude']), float(row['longitude'])
    if not (SOUTH <= lat <= NORTH and WEST <= lon <= EAST):
        return None, 'outside_roi'
    dt = datetime.fromisoformat(row['base_date_time'].strip().replace('Z', '+00:00'))
    dt = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    if dt.year != YEAR:
        return None, 'outside_year'
    sog, cog = float(row['sog']), float(row['cog'])
    if not (0 <= sog <= SOG_MAX and 0 <= cog < 360):
        return None, 'invalid_sog_cog'
    mmsi = int(row['mmsi'])
    if not 1 <= mmsi <= 999999999:
        return None, 'invalid_mmsi'
    if cargo_only:
        ship_type = optional_number(row.get('vessel_type'))
        if ship_type is None or not (70 <= ship_type < 90):
            return None, 'not_cargo_tanker'
    heading = optional_number(row.get('heading'))
    if heading is not None and not 0 <= heading < 360:
        heading = None
    status = optional_number(row.get('status'))
    if status == 15:  # AIS not-defined navigation status
        status = None
    return (lat, lon, sog, cog, heading, optional_number(row.get('rot')),
            status, dt.timestamp(), mmsi), 'kept'



SCHEMA = 'CREATE TABLE messages (lat REAL, lon REAL, sog REAL, cog REAL, heading REAL, rot REAL, status REAL, ts REAL, mmsi INTEGER)'
INSERT = 'INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?)'
_PROGRESS = None
_STOP = None


def init_worker(progress, stop):
    global _PROGRESS, _STOP
    _PROGRESS, _STOP = progress, stop


class IndexedRow:
    """Expose only requested CSV fields without building a per-row dictionary."""
    def __init__(self, fields):
        self.indices = {name: i for i, name in enumerate(fields)}
        self.values = []

    def __getitem__(self, name):
        index = self.indices[name]
        return self.values[index] if index < len(self.values) else None

    def get(self, name):
        return self[name] if name in self.indices else None


def read_file(index, path, database, cargo_only, batch_size):
    started = time.monotonic()
    counts = Counter()
    sent = Counter()
    last_report = started
    connection = sqlite3.connect(database)
    try:
        connection.execute(SCHEMA)
        batch = []
        with path.open('rb') as raw:
            binary = gzip.GzipFile(fileobj=raw) if path.name.lower().endswith('.gz') else raw
            with io.TextIOWrapper(binary, encoding='utf-8-sig', newline='') as stream:
                reader = csv.reader(stream)
                fields = next(reader, None)
                if fields is None:
                    raise ValueError(f'{path.name}: missing CSV header')
                fields = [name.strip().lower() for name in fields]
                missing = REQUIRED - set(fields)
                if cargo_only and 'vessel_type' not in fields:
                    missing.add('vessel_type')
                if missing:
                    raise ValueError(f'{path.name}: missing columns {sorted(missing)}')
                row = IndexedRow(fields)
                for values in reader:
                    if not values:  # Match DictReader's empty-line handling.
                        continue
                    row.values = values
                    counts['rows_read'] += 1
                    try:
                        parsed, reason = parse_row(row, cargo_only)
                    except (ValueError, TypeError, OverflowError, AttributeError):
                        parsed, reason = None, 'malformed_required_field'
                    counts[reason] += 1
                    if parsed is not None:
                        batch.append(parsed)
                    if len(batch) >= batch_size:
                        connection.executemany(INSERT, batch)
                        connection.commit()
                        batch.clear()
                    if counts['rows_read'] % 10000 == 0:
                        if _STOP.is_set():
                            raise RuntimeError('Conversion cancelled')
                        now = time.monotonic()
                        if now - last_report >= 0.5:
                            _PROGRESS.put((index, raw.tell(), dict(counts - sent)))
                            sent = counts.copy()
                            last_report = now
                if batch:
                    connection.executemany(INSERT, batch)
                    connection.commit()
        _PROGRESS.put((index, path.stat().st_size, dict(counts - sent)))
        elapsed = time.monotonic() - started
        return {'file': path.name, 'seconds': elapsed, 'counts': dict(counts),
                'rows_per_second': counts['rows_read']/max(elapsed, 1e-6)}
    finally:
        connection.close()


def read_parallel(files, temporary, connection, args, monitor):
    """Bound pending files, merge in input order, and keep timestamp ties stable."""
    context = mp.get_context('spawn')  # Safe on macOS, including Apple Silicon.
    progress = context.Queue()
    stop = context.Event()
    workers = min(args.workers, len(files))
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=context,
                                   initializer=init_worker, initargs=(progress, stop))
    pending = {}
    all_futures = []
    positions = {}
    file_stats = []
    final_counts = Counter()

    def drain():
        while True:
            try:
                index, position, delta = progress.get_nowait()
            except queue.Empty:
                break
            with monitor.lock:
                previous = positions.get(index, 0)
                positions[index] = position
                monitor.completed_bytes += position - previous
                monitor.counts.update(delta)

    def submit(index):
        database = str(Path(temporary) / f'part_{index:06d}.sqlite')
        pending[index] = (executor.submit(read_file, index, files[index], database,
                                         args.cargo_tanker_only, args.batch_size), database)
        all_futures.append(pending[index][0])

    monitor.log(f'Using {workers} worker processes; batch size {args.batch_size:,}')
    try:
        for index in range(workers):
            submit(index)
        for index, path in enumerate(files):
            future, database = pending.pop(index)
            with monitor.lock:
                monitor.file_label = f'{index}/{len(files)} files merged; {workers} workers'
            while not future.done():
                drain()
                stop.wait(0.1)
            stats = future.result()
            drain()
            monitor.log(f'[{index+1}/{len(files)}] Merging {path.name}; parsed in {stats["seconds"]:.1f}s')
            merge_started = time.monotonic()
            connection.execute('ATTACH DATABASE ? AS part', (database,))
            try:
                connection.execute('INSERT INTO messages SELECT * FROM part.messages ORDER BY rowid')
                connection.commit()
            finally:
                connection.execute('DETACH DATABASE part')
            Path(database).unlink()
            stats['merge_seconds'] = time.monotonic() - merge_started
            file_stats.append(stats)
            final_counts.update(stats['counts'])
            monitor.log(f'  Finished: {stats["counts"]}')
            next_index = index + workers
            if next_index < len(files):
                submit(next_index)
    finally:
        stop.set()
        # Drain progress while workers stop, preventing queue feeder deadlocks.
        executor.shutdown(wait=False, cancel_futures=True)
        for future in all_futures:
            while not future.done():
                drain()
                time.sleep(0.05)
        executor.shutdown(wait=True)
        drain()
        progress.close()
        progress.join_thread()
    with monitor.lock:
        monitor.counts.clear()
        monitor.counts.update(final_counts)
        monitor.completed_bytes = monitor.total_bytes
        monitor.file_label = f'{len(files)}/{len(files)} files merged'
    return file_stats



def mean_encoding(args):
    resolutions = [args.mean_lat_resolution, args.mean_lon_resolution,
                   args.mean_sog_resolution, args.mean_cog_resolution]
    spans = np.array([NORTH-SOUTH, EAST-WEST, SOG_MAX, 360.0])
    if any(not math.isfinite(x) or x <= 0 for x in resolutions):
        raise ValueError('Mean resolutions must be finite and positive')
    bins = np.array([math.ceil(span/res) for span, res in zip(spans, resolutions)], dtype=np.int64)
    if bins.sum() > 10000000:
        raise ValueError('Requested mean vector is too large; use coarser resolutions')
    return spans, bins, resolutions


def save_mean(tracks, output, args, monitor=None, normalized=False, source=None):
    """Same bin assignment as calculate_AIS_mean.py, using counts not dense vectors."""
    if not isinstance(tracks, dict):
        raise ValueError('Training pickle must contain a dictionary of track arrays')
    spans, bins, resolutions = mean_encoding(args)
    total = sum(len(a) for a in tracks.values())
    if total == 0:
        raise ValueError('Cannot compute mean.pkl from an empty training set')
    if monitor:
        monitor.set_phase('calculating training four-hot mean', total)
    report = monitor.log if monitor else lambda message: print(message, flush=True)
    report(f'Calculating mean from {total:,} training messages; bins={bins.tolist()}')
    counts = [np.zeros(int(n), dtype=np.int64) for n in bins]
    minimum = np.array([SOUTH, WEST, 0.0, 0.0])
    seen = 0
    last_report = time.monotonic()
    for key, array in tracks.items():
        array = np.asarray(array)
        if array.ndim != 2 or array.shape[1] < 4:
            raise ValueError(f'Track {key}: expected array with at least four columns')
        for start in range(0, len(array), 100000):
            values = np.array(array[start:start+100000, :4], dtype=np.float64, copy=True)
            if not np.isfinite(values).all():
                raise ValueError(f'Track {key}: nonfinite latitude/longitude/SOG/COG')
            if not normalized:
                values = (values-minimum)/spans
            if np.any(values < 0) or np.any(values > 1):
                raise ValueError(f'Track {key}: values outside encoding range; check ROI and --mean-input-normalized')
            # Match the supplied reference's treatment of an exact upper bound.
            values[values == 1] = 0.99999
            indices = (values*bins).astype(np.int64)
            for column, n in enumerate(bins):
                counts[column] += np.bincount(indices[:,column], minlength=int(n))
            seen += len(values)
            if monitor:
                monitor.done = seen
            elif time.monotonic()-last_report >= args.progress_seconds:
                report(f'Mean: {seen:,}/{total:,} training messages')
                last_report = time.monotonic()
    mean = np.concatenate(counts).astype(np.float64)/seen
    output.mkdir(parents=True, exist_ok=True)
    target = output/'mean.pkl'
    # Atomic replacement: an interrupted write does not damage an existing mean.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output, prefix='.mean-', delete=False) as stream:
            temporary = Path(stream.name)
            pickle.dump(mean, stream, protocol=4)
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    information = {
        'source_training_pickle': source, 'training_messages': seen,
        'encoding': 'concatenated four-hot frequency vector',
        'columns': ['latitude','longitude','sog','cog'],
        'bins': bins.tolist(), 'resolutions': resolutions, 'data_dim': int(bins.sum()),
        'roi': {'south': SOUTH,'north': NORTH,'west': WEST,'east': EAST},
        'sog_max': SOG_MAX, 'input_normalized': normalized,
        'upper_boundary_rule': 'normalized values equal to 1 become 0.99999',
        'training_preprocessing_note': 'Recompute after changing training observations',
    }
    (output/'mean_metadata.json').write_text(json.dumps(information, indent=2)+'\n')
    report(f'Saved {target}: {len(mean)} values; four block sums = {[float(c.sum()/seen) for c in counts]}')
    return information


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input-dir', type=Path)
    parser.add_argument('--train', type=float, default=80, help='Training percentage (default 80)')
    parser.add_argument('--valid', type=float, default=10, help='Validation percentage (default 10)')
    parser.add_argument('--test', type=float, default=10, help='Test percentage (default 10)')
    parser.add_argument('--output-dir', type=Path, help='Optional output folder; must not already exist')
    parser.add_argument('--cargo-tanker-only', action='store_true', help='Keep rows with vessel_type 70–89; default keeps all vessel types')
    parser.add_argument('--progress-seconds', type=float, default=5,
                        help='Monitoring interval in seconds (default 5)')
    parser.add_argument('--workers', type=int, default=2,
                        help='Parallel file-reading processes (default 2; try 4 on M1)')
    parser.add_argument('--batch-size', type=int, default=50000,
                        help='Retained rows per worker SQLite transaction (default 50000)')
    parser.add_argument('--mean-only', type=Path, help='Create mean.pkl from an existing TRAINING pickle, without reading CSVs')
    parser.add_argument('--mean-input-normalized', action='store_true', help='With --mean-only: first four columns are already in [0,1]')
    parser.add_argument('--mean-lat-resolution', type=float, default=0.01)
    parser.add_argument('--mean-lon-resolution', type=float, default=0.01)
    parser.add_argument('--mean-sog-resolution', type=float, default=1.0)
    parser.add_argument('--mean-cog-resolution', type=float, default=5.0)
    args = parser.parse_args()
    if args.workers < 1 or args.batch_size < 1:
        parser.error('--workers and --batch-size must be positive integers')
    if not math.isfinite(args.progress_seconds) or args.progress_seconds <= 0:
        parser.error('--progress-seconds must be positive and finite')
    try:
        mean_encoding(args)
    except (ValueError, OverflowError) as exc:
        parser.error(str(exc))
    if args.mean_only:
        if args.input_dir:
            parser.error('Use either --mean-only or --input-dir')
        print(f'Loading training pickle {args.mean_only}...', flush=True)
        with args.mean_only.open('rb') as stream:
            tracks = pickle.load(stream, encoding='latin1')
        save_mean(tracks, args.output_dir or args.mean_only.parent, args,
                  normalized=args.mean_input_normalized, source=str(args.mean_only.resolve()))
        return
    if args.mean_input_normalized:
        parser.error('--mean-input-normalized requires --mean-only')
    if args.input_dir is None:
        parser.error('--input-dir is required unless --mean-only is used')
    percentages = [args.train, args.valid, args.test]
    if any(not math.isfinite(p) or p < 0 for p in percentages) or not math.isclose(sum(percentages), 100, abs_tol=1e-8):
        parser.error('--train, --valid and --test must be nonnegative and sum to 100')
    if args.train <= 0:
        parser.error('--train must be positive to calculate mean.pkl')
    if not args.input_dir.is_dir():
        parser.error(f'Input directory not found: {args.input_dir.resolve()}')
    files = sorted(p for p in args.input_dir.iterdir() if p.is_file() and (p.name.lower().endswith('.csv') or p.name.lower().endswith('.csv.gz')))
    if not files:
        parser.error('No .csv or .csv.gz files found in the input directory')
    # Avoid reading both a compressed file and its decompressed copy.
    names = {p.name.lower() for p in files}
    duplicates = [p.name for p in files if p.name.lower().endswith('.gz') and p.name.lower()[:-3] in names]
    if duplicates:
        parser.error(f'Both compressed and plain copies found; keep only one copy: {duplicates}')
    label = 'cargo_tanker' if args.cargo_tanker_only else 'all_vessels'
    output = args.output_dir or Path(f'ais_2021_roi_S{SOUTH}_N{NORTH}_W{WEST}_E{EAST}_{label}_train{args.train:g}_valid{args.valid:g}_test{args.test:g}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}')
    output.mkdir(parents=True, exist_ok=False)
    counts = Counter()
    started = time.monotonic()
    monitor = Monitor(output, counts, sum(p.stat().st_size for p in files), args.progress_seconds)
    report = monitor.log
    file_stats = []
    report('Reading ETA is approximate, based on input bytes; excludes indexing/export.')
    if psutil is None:
        report('Optional RAM monitoring: python3 -m pip install psutil')
    try:
        with tempfile.TemporaryDirectory(prefix='.staging_', dir=output) as temporary:
            connection = sqlite3.connect(str(Path(temporary) / 'messages.sqlite'))
            try:
                connection.execute(SCHEMA)
                monitor.set_phase('reading')
                file_stats = read_parallel(files, temporary, connection, args, monitor)
                total = counts['kept']
                if not total:
                    raise ValueError('No valid 2021 messages inside the ROI. See filtering counts.')
                monitor.set_phase('indexing timestamps')
                connection.execute('CREATE INDEX time_index ON messages(ts)')
                def boundary(fraction):
                    offset = min(total, max(0, int(total * fraction / 100)))
                    if offset == total:
                        return float('inf')
                    return connection.execute('SELECT ts FROM messages ORDER BY ts LIMIT 1 OFFSET ?', (offset,)).fetchone()[0]
                monitor.set_phase('finding split boundaries')
                cuts = [float('-inf'), boundary(args.train), boundary(args.train + args.valid), float('inf')]
                split_stats = {}
                for name, lower, upper in zip(['train', 'valid', 'test'], cuts, cuts[1:]):
                    monitor.set_phase(f'counting {name} vessel messages')
                    # Preallocate arrays to avoid costly concatenations and Python row lists.
                    sizes = connection.execute('SELECT mmsi, COUNT(*) FROM messages WHERE ts >= ? AND ts < ? GROUP BY mmsi', (lower, upper)).fetchall()
                    n_messages = sum(size for _, size in sizes)
                    report(f'{name}: arrays require at least {n_messages*9*8/2**30:.3f} GiB, plus Python/SQLite overhead')
                    monitor.set_phase(f'allocating {name} arrays')
                    tracks = {mmsi: np.empty((size, 9), dtype=np.float64) for mmsi, size in sizes}
                    positions = dict.fromkeys(tracks, 0)
                    n_messages = sum(size for _, size in sizes)
                    monitor.set_phase(f'building {name} tracks', n_messages)
                    for row in connection.execute('SELECT * FROM messages WHERE ts >= ? AND ts < ? ORDER BY ts', (lower, upper)):
                        mmsi = row[8]
                        tracks[mmsi][positions[mmsi]] = row
                        positions[mmsi] += 1
                        monitor.done += 1
                    monitor.set_phase(f'saving {name} pickle')
                    with (output / f'{name}_tracks.pkl').open('wb') as stream:
                        pickle.dump(tracks, stream, protocol=4)
                    split_stats[name] = {'messages': n_messages, 'vessels': len(tracks), 'actual_percentage': 100*n_messages/total}
                    report(f'  Saved {n_messages:,} messages / {len(tracks):,} vessels', flush=True)
                    if name == 'train':
                        mean_info = save_mean(tracks, output, args, monitor,
                                              source=str((output/'train_tracks.pkl').resolve()))
                    del tracks, positions
                monitor.set_phase('writing metadata')
                metadata = {
                    'mean': mean_info,
                    'monitoring': {'file_statistics': file_stats,
                                   'completed_stage_seconds': dict(monitor.phase_seconds),
                                   'peak_sampled_rss_bytes': monitor.peak_memory if psutil else None,
                                   'log_file': 'run.log'},
                    'input_files': [str(p.resolve()) for p in files], 'year': YEAR,
                    'workers': min(args.workers, len(files)), 'batch_size': args.batch_size,
                    'roi': {'south': SOUTH, 'north': NORTH, 'west': WEST, 'east': EAST},
                    'requested_percentages': dict(zip(['train', 'valid', 'test'], percentages)),
                    'split_method': 'Chronological retained-message percentages; timestamp ties go to the later split',
                    'split_boundaries_utc': [datetime.fromtimestamp(t, timezone.utc).isoformat() if math.isfinite(t) else None for t in cuts[1:3]],
                    'columns': COLUMNS, 'missing_optional_values': 'NaN', 'timestamps': 'Unix seconds UTC',
                    'cargo_tanker_only': args.cargo_tanker_only, 'sog_max_knots': SOG_MAX,
                    'distance_to_coast_filter': False, 'counts': dict(counts), 'splits': split_stats,
                    'elapsed_seconds': time.monotonic() - started,
                }
                (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
            finally:
                connection.close()
    except (Exception, KeyboardInterrupt):
        report(f'Conversion failed. Partial outputs may remain in {output.resolve()}', flush=True)
        report(f'Filtering counts: {dict(counts)}', flush=True)
        monitor.close()
        raise
    report(f'Filtering counts: {dict(counts)}', flush=True)
    report(f'Complete in {duration(time.monotonic()-started)}: {output.resolve()}', flush=True)
    monitor.close()


if __name__ == '__main__':
    main()