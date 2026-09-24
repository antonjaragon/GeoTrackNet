#!/usr/bin/env python3
"""Prepare csv2pkl outputs for GeoTrackNet using its original utils functions.

Run from the GeoTrackNet repository root (where utils.py is located):
  python dataset_preprocessing.py --input-dir data/ais_ct_2021_florida_straits_70_15_15

Reads train_tracks.pkl, valid_tracks.pkl, test_tracks.pkl sequentially. Writes
normalized train.pkl, valid.pkl, test.pkl, mean.pkl and metadata.json into a NEW
sibling folder ending in _processed. Raw inputs are never modified.
Requires NumPy and the repository's utils.py plus its dependencies.
--utils-dir selects another repository root. Only load trusted pickle files.

Retains original thresholds: gaps >2h split voyages; >=20 source observations
and >=4h duration; original detectOutlier/interpolate; 300-second samples;
max 288 samples/segment, min 48 samples (original sample-count convention);
remove >70% anchored/moored, max speed <1 knot, or >80% speeds <2 knots.
Sorts timestamps and keeps the first observation at each duplicate timestamp.
After outlier removal, splits again at gaps >2h to avoid interpolating new gaps.
Unknown optional fields remain NaN and do not cause row rejection here.
Errors from repository utils fail with context instead of silently deleting data.

Output dictionary keys identify segments, NOT MMSIs; MMSI remains column 8.
First four columns are normalized to [0,1]. Other five columns retain units.
Mean uses all processed training messages, matching calculate_AIS_mean.py.
The provided datasets.py subsequently takes every second sample (10 minutes).
Each raw split and its processed output must fit in RAM.
"""
import argparse
import ast
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import pickle
import sys
import time
import numpy as np

LAT, LON, SOG, COG, HEADING, ROT, STATUS, TS, MMSI = range(9)


def load_utils(directory):
    path = directory.resolve() / 'utils.py'
    if not path.is_file():
        raise FileNotFoundError(f'{path} not found. Run from GeoTrackNet root or supply --utils-dir.')
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('geotracknet_preprocessing_utils', path)
    module = importlib.util.module_from_spec(spec)
    # Adapt removed scalar aliases in this legacy module only; do not modify NumPy.
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    class LegacyAliases(ast.NodeTransformer):
        def visit_Attribute(self, node):
            self.generic_visit(node)
            if (isinstance(node.value, ast.Name) and node.value.id in ('np', 'numpy')
                    and node.attr in ('float', 'int', 'complex', 'bool', 'object', 'str')):
                return ast.copy_location(ast.Name(id=node.attr, ctx=node.ctx), node)
            return node
    tree = ast.fix_missing_locations(LegacyAliases().visit(tree))
    exec(compile(tree, str(path), 'exec'), module.__dict__)
    for name in ('detectOutlier', 'interpolate'):
        if not callable(getattr(module, name, None)):
            raise ValueError(f'{path} must define {name}')
    return module


def split_gaps(v):
    return np.split(v, np.flatnonzero(np.diff(v[:, TS]) > 7200) + 1)


def remove_outliers(v, utils):
    reported, calculated = utils.detectOutlier(v[:, [TS,LAT,LON,SOG]], speed_max=30)
    reported, calculated = np.asarray(reported, dtype=bool), np.asarray(calculated, dtype=bool)
    if reported.shape != (len(v),):
        raise ValueError('detectOutlier returned an unexpected reported mask shape')
    # Original code applies the calculated mask AFTER the reported mask.
    if reported.all() or (calculated.size and calculated.all()):
        return v[:0]
    remaining = v[~reported]
    if calculated.shape != (len(remaining),):
        raise ValueError('detectOutlier calculated mask must match rows remaining after reported mask')
    return remaining[~calculated]


def process(raw, args, utils, split):
    if not isinstance(raw, dict):
        raise ValueError('Input must be an MMSI -> (N,9) array dictionary')
    result, stats = {}, Counter()
    last = time.monotonic()
    for index, key in enumerate(list(raw), 1):
        v = np.asarray(raw.pop(key), dtype=np.float64)
        if v.ndim != 2 or v.shape[1] != 9:
            raise ValueError(f'{split}, vessel {key}: expected (N,9) array')
        stats['input_messages'] += len(v)
        valid = np.isfinite(v[:,[LAT,LON,SOG,COG,TS,MMSI]]).all(axis=1)
        valid &= (v[:,LAT]>=args.lat_min)&(v[:,LAT]<=args.lat_max)
        valid &= (v[:,LON]>=args.lon_min)&(v[:,LON]<=args.lon_max)
        valid &= (v[:,SOG]>=0)&(v[:,SOG]<=30)&(v[:,COG]>=0)&(v[:,COG]<360)
        stats['invalid_or_outside_rows'] += int((~valid).sum())
        v = v[valid]
        if len(v):
            v = v[np.argsort(v[:,TS], kind='stable')]
            unique = np.r_[True,np.diff(v[:,TS])>0]
            stats['duplicate_timestamps'] += int((~unique).sum())
            v = v[unique]
        if not len(v):
            continue
        for voyage in split_gaps(v):
            if len(voyage)<20 or voyage[-1,TS]-voyage[0,TS]<14400:
                stats['short_source_voyages'] += 1
                continue
            try:
                clean = remove_outliers(voyage, utils)
            except Exception as exc:
                raise RuntimeError(f'{split}, vessel {key}, detectOutlier: {exc}') from exc
            stats['outlier_rows'] += len(voyage)-len(clean)
            if not len(clean):
                continue
            for contiguous in split_gaps(clean):
                if len(contiguous)<2 or contiguous[-1,TS]-contiguous[0,TS]<14400:
                    stats['short_after_outliers'] += 1
                    continue
                sampled = []
                for timestamp in range(int(contiguous[0,TS]),int(contiguous[-1,TS]),300):
                    try:
                        row = utils.interpolate(timestamp, contiguous)
                    except Exception as exc:
                        raise RuntimeError(f'{split}, vessel {key}, interpolate at {timestamp}: {exc}') from exc
                    if row is None:
                        sampled = []
                        stats['interpolation_rejected_voyages'] += 1
                        break
                    row = np.asarray(row,dtype=np.float64).reshape(-1)
                    if row.shape != (9,) or not np.isfinite(row[[LAT,LON,SOG,COG,TS,MMSI]]).all():
                        raise ValueError(f'{split}, vessel {key}: interpolation returned invalid required fields')
                    sampled.append(row)
                if not sampled:
                    continue
                sampled = np.asarray(sampled)
                if not np.allclose(np.diff(sampled[:,TS]),300,rtol=0,atol=1e-6):
                    raise ValueError('Interpolation did not produce 300-second intervals')
                for start in range(0,len(sampled),288):
                    segment = sampled[start:start+288].copy()
                    if len(segment)<48:
                        stats['short_sampled_segments'] += 1
                        continue
                    if (np.mean(segment[:,STATUS]==1)>.7 or np.mean(segment[:,STATUS]==5)>.7
                            or segment[:,SOG].max()<1 or np.mean(segment[:,SOG]<2)>.8):
                        stats['stationary_or_slow_segments'] += 1
                        continue
                    segment[:,LAT] = (segment[:,LAT]-args.lat_min)/(args.lat_max-args.lat_min)
                    segment[:,LON] = (segment[:,LON]-args.lon_min)/(args.lon_max-args.lon_min)
                    segment[:,SOG] /= 30
                    segment[:,COG] /= 360
                    if np.any(segment[:,:4]<-1e-10) or np.any(segment[:,:4]>1+1e-10):
                        raise ValueError('Interpolated fields outside normalization range')
                    segment[:,:4] = np.clip(segment[:,:4],0,1)
                    result[len(result)] = segment
        if time.monotonic()-last>=5:
            print(f'{split}: {index:,} vessels processed; {len(result):,} segments retained',flush=True)
            last=time.monotonic()
    stats['output_segments']=len(result)
    stats['output_messages']=sum(len(v) for v in result.values())
    return result,dict(stats)


def calculate_mean(data,bins):
    counts=[np.zeros(n,dtype=np.int64) for n in bins]
    total=0
    for v in data.values():
        x=v[:,:4].copy()
        x[x==1]=.99999
        idx=(x*np.asarray(bins)).astype(np.int64)
        for col,n in enumerate(bins):
            counts[col]+=np.bincount(idx[:,col],minlength=n)
        total+=len(v)
    if not total:
        raise ValueError('No processed training messages survived; cannot create mean.pkl')
    return np.concatenate(counts).astype(np.float64)/total


def dump(path,data):
    with path.open('wb') as f:
        pickle.dump(data,f,protocol=4)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input-dir','--dataset_dir',dest='input_dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--utils-dir',type=Path,default=Path.cwd())
    for name,value in [('lat_min',24.1055773),('lat_max',26.7286216),('lon_min',-81.2981022),('lon_max',-77.262825)]:
        parser.add_argument('--'+name,type=float,default=value)
    for name,value in [('lat',.01),('lon',.01),('sog',1.),('cog',5.)]:
        parser.add_argument('--onehot_'+name+'_reso',type=float,default=value)
    args=parser.parse_args()
    spans=[args.lat_max-args.lat_min,args.lon_max-args.lon_min,30.,360.]
    res=[getattr(args,'onehot_'+n+'_reso') for n in ('lat','lon','sog','cog')]
    if any(not math.isfinite(x) or x<=0 for x in spans+res):
        parser.error('ROI ranges and resolutions must be positive and finite')
    bins=[math.ceil(s/r) for s,r in zip(spans,res)]
    if sum(bins)>10000000:
        parser.error('Encoding too large; increase resolutions')
    for split in ('train','valid','test'):
        if not (args.input_dir/f'{split}_tracks.pkl').is_file():
            parser.error(f'Missing {split}_tracks.pkl in {args.input_dir}')
    utils=load_utils(args.utils_dir)
    output=args.output_dir or args.input_dir.resolve().with_name(args.input_dir.resolve().name+'_processed')
    output.mkdir(parents=True,exist_ok=False)
    metadata={'normalized':True,'sample_interval_seconds':300,'bins':bins,'data_dim':sum(bins),
              'roi':{'south':args.lat_min,'north':args.lat_max,'west':args.lon_min,'east':args.lon_max},
              'resolutions':res,'mean_source':'all processed training messages',
              'utils_file':str((args.utils_dir/'utils.py').resolve()),'splits':{}}
    for split in ('train','valid','test'):
        print(f'Loading and preprocessing {split}...',flush=True)
        with (args.input_dir/f'{split}_tracks.pkl').open('rb') as f:
            raw=pickle.load(f,encoding='latin1')
        processed,stats=process(raw,args,utils,split)
        metadata['splits'][split]=stats
        print(f'{split}: {stats}',flush=True)
        if split=='train':
            dump(output/'mean.pkl',calculate_mean(processed,bins))
        if not processed:
            print(f'WARNING: {split} is empty after preprocessing.',flush=True)
        dump(output/f'{split}.pkl',processed)
        del raw,processed
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'Complete: {output}\nFour-hot dimensions: {bins} = {sum(bins)}',flush=True)


if __name__=='__main__':
    main()
