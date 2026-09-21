#!/usr/bin/env python3
"""Download daily NOAA AIS .csv.zst files (Python 3.8+, no extra packages).

Examples:
  python3 download_ais.py
  python3 download_ais.py --estimate-only
  python3 download_ais.py --start 2021-01-01 --end 2021-01-31 --output january_ais

Files stay compressed. Size estimates do not include decompressed CSVs.
Completed files of the expected size are skipped on subsequent runs. Failed
files are retried from the beginning; .part files are never treated as complete.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
import shutil
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_URL = 'https://noaaocm.blob.core.windows.net/ais/csv2'
TIMEOUT = 60
ATTEMPTS = 3
CHUNK = 1024 * 1024


def human_size(size):
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if size < 1024 or unit == 'TiB':
            return '{:,.2f} {}'.format(size, unit)
        size /= 1024


def file_url(day):
    return '{}/csv{}/ais-{}.csv.zst'.format(BASE_URL, day.year, day.isoformat())


def request(url, method='GET'):
    return Request(url, method=method, headers={
        'User-Agent': 'AIS-Bulk-Downloader/1.0', 'Accept-Encoding': 'identity'})


def inspect_file(day):
    """Read headers only; return (day, byte size or None, error or None)."""
    error = None
    for attempt in range(ATTEMPTS):
        try:
            with urlopen(request(file_url(day), 'HEAD'), timeout=TIMEOUT) as response:
                length = response.headers.get('Content-Length')
                return day, int(length) if length is not None else None, None
        except (HTTPError, URLError, OSError, ValueError) as exc:
            error = str(exc)
            if isinstance(exc, HTTPError) and exc.code in (403, 404):
                break
            if attempt < ATTEMPTS - 1:
                time.sleep(2 ** attempt)
    return day, None, error


def download(day, expected_size, folder):
    destination = folder / ('ais-{}.csv.zst'.format(day.isoformat()))
    partial = destination.with_name(destination.name + '.part')
    for attempt in range(1, ATTEMPTS + 1):
        try:
            with urlopen(request(file_url(day)), timeout=TIMEOUT) as response:
                length = response.headers.get('Content-Length')
                size = int(length) if length is not None else expected_size
                written = 0
                last_update = time.monotonic()
                with partial.open('wb') as output:
                    while True:
                        chunk = response.read(CHUNK)
                        if not chunk:
                            break
                        output.write(chunk)
                        written += len(chunk)
                        if time.monotonic() - last_update >= 2:
                            suffix = ' / ' + human_size(size) if size is not None else ''
                            print('\r  {}{} downloaded'.format(human_size(written), suffix),
                                  end='', flush=True)
                            last_update = time.monotonic()
                if size is not None and written != size:
                    raise OSError('Incomplete download: {} of {} bytes'.format(written, size))
            partial.replace(destination)
            print('\r  Saved {} ({})'.format(destination.name, human_size(written)))
            return True
        except (HTTPError, URLError, OSError, ValueError) as exc:
            print('\n  Attempt {}/{} failed: {}'.format(attempt, ATTEMPTS, exc))
            if isinstance(exc, HTTPError) and exc.code in (403, 404):
                break
            if attempt < ATTEMPTS:
                time.sleep(2 ** (attempt - 1))
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--start', type=date.fromisoformat, default=date(2021, 1, 1))
    parser.add_argument('--end', type=date.fromisoformat, default=date(2021, 12, 31))
    parser.add_argument('--output', type=Path, help='Destination folder (default: ais_2021 for 2021)')
    parser.add_argument('--estimate-only', action='store_true', help='Check sizes without downloading')
    parser.add_argument('--yes', action='store_true', help='Download without the confirmation prompt')
    args = parser.parse_args()
    if args.end < args.start:
        parser.error('--end must be on or after --start')
    days = [args.start + timedelta(days=n) for n in range((args.end - args.start).days + 1)]
    label = str(args.start.year) if args.start.year == args.end.year else '{}_to_{}'.format(args.start, args.end)
    folder = (args.output or Path('ais_' + label)).expanduser().resolve()
    print('Checking sizes for {} daily files (headers only)...'.format(len(days)), flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(inspect_file, days))
    known = [size for _, size, _ in records if size is not None]
    unknown = len(records) - len(known)
    total_known = sum(known)
    average = total_known / len(known) if known else None
    pending = []
    for day, size, error in records:
        path = folder / ('ais-{}.csv.zst'.format(day.isoformat()))
        if size is not None and path.is_file() and path.stat().st_size == size:
            continue
        pending.append((day, size))
    print('Folder: {}'.format(folder))
    print('Compressed size reported by server: {} ({} of {} files)'.format(
        human_size(total_known), len(known), len(records)))
    if unknown:
        print('{} files have unknown sizes or unavailable headers.'.format(unknown))
        if average is not None:
            print('Estimated full selection: {} (unknown sizes use the known-file average)'.format(
                human_size(total_known + unknown * average)))
        else:
            print('Total size cannot be estimated: no file sizes were available.')
    print('Already downloaded (matching size): {}'.format(len(records) - len(pending)))
    remaining = sum(size for _, size in pending if size is not None)
    missing_sizes = sum(size is None for _, size in pending)
    estimate = remaining + missing_sizes * average if average is not None else None
    if estimate is not None:
        print('{}download remaining: {}'.format('Estimated ' if missing_sizes else '', human_size(estimate)))
    print('Files remain compressed; decompressed CSVs require additional space.')
    if args.estimate_only or not pending:
        return 0
    folder.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(folder).free
    print('Free disk space: {}'.format(human_size(free)))
    if remaining > free:
        print('Insufficient free space for even the known-size downloads.', file=sys.stderr)
        return 1
    if estimate is not None and estimate > free:
        print('Warning: estimated downloads exceed free disk space.')
    if not args.yes:
        try:
            answer = input('Download {} files? [y/N] '.format(len(pending)))
        except EOFError:
            print('No confirmation available. Use --yes for unattended downloads.')
            return 1
        if answer.strip().lower() not in ('y', 'yes'):
            print('Cancelled.')
            return 0
    failed = []
    for index, (day, size) in enumerate(pending, 1):
        print('[{}/{}] {}'.format(index, len(pending), day), flush=True)
        if not download(day, size, folder):
            failed.append(day.isoformat())
    print('\nFinished: {} downloaded, {} failed.'.format(len(pending) - len(failed), len(failed)))
    if failed:
        print('Failed dates: ' + ', '.join(failed))
        print('Run the same command again to retry; completed files are skipped.')
    return 1 if failed else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('\nStopped. Run again to skip completed files and retry unfinished ones.')
        sys.exit(130)
