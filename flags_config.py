# coding: utf-8

# MIT License
# 
# Copyright (c) 2018 Duong Nguyen
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

"""
Flag configuration.
Adapted from the original script of FIVO.
"""

import os
import tensorflow as tf
import pickle
import math
import json
import sys
import numpy as np


## Bretagne dataset
# LAT_MIN = 46.5
# LAT_MAX = 50.5
# LON_MIN = -8.0
# LON_MAX = -3.0

# ## Aruba
# LAT_MIN = 11.0
# LAT_MAX = 14.0
# LON_MIN = -71.0
# LON_MAX = -68.0

## Gulf of Mexico
"""
LAT_MIN = 26.5
LAT_MAX = 30.0
LON_MIN = -97.5
LON_MAX = -87
"""

SPEED_MAX = 30.0  # knots
FIG_DPI = 150

# Shared flags.
tf.app.flags.DEFINE_string("mode", "train",
                           "The mode of the binary. Must be 'train'"
                           "'save_logprob','local_logprob'"
                           "'contrario_detection','visualisation'"
                           "'traj_reconstruction' or 'traj_speed'.")

tf.app.flags.DEFINE_string("bound", "elbo",
                           "The bound to optimize. Can be 'elbo', or 'fivo'.")

tf.app.flags.DEFINE_integer("latent_size", 64,
                            "The size of the latent state of the model.")

tf.app.flags.DEFINE_string("log_dir", "./chkpt",
                           "The directory to keep checkpoints and summaries in.")

tf.app.flags.DEFINE_integer("batch_size", 32,
                            "Batch size.")
tf.app.flags.DEFINE_integer("num_samples", 16,
                           "The number of samples (or particles) for multisample "
                           "algorithms.")
tf.app.flags.DEFINE_float("ll_thresh", -17.47,
                          "Log likelihood for the anomaly detection.")


# Dataset flags
tf.app.flags.DEFINE_string("dataset_dir", "./data",
                           "Dataset directory")
tf.app.flags.DEFINE_string("trainingset_name", "ct_aruba_2019/ct_aruba_2019_train.pkl",
                           "Path to load the trainingset from.")
tf.app.flags.DEFINE_string("testset_name", "ct_aruba_2019/ct_aruba_2019_test.pkl",
                           "Path to load the testset from.")
tf.app.flags.DEFINE_string("split", "train",
                           "Split to evaluate the model on. Can be 'train', 'valid', or 'test'.")
tf.app.flags.DEFINE_boolean("missing_data", False,
                           "If true, a part of input track will be deleted.")


# Model flags
tf.app.flags.DEFINE_string("model", "vrnn",
                           "Model choice. Currently only 'vrnn' is supported.")

tf.app.flags.DEFINE_integer("random_seed", None,
                            "A random seed for seeding the TensorFlow graph.")


# Track flags.
tf.app.flags.DEFINE_float("interval_max", 2*3600,
                          "Maximum interval between two successive AIS messages (in second).")
tf.app.flags.DEFINE_integer("min_duration", 4,
                            "Min duration (hour) of a vessel track")

# Four-hot-encoding flags.
tf.app.flags.DEFINE_float("lat_min", 11.0,
                          "ROI")
tf.app.flags.DEFINE_float("lat_max", 14.0,
                          "ROI")
tf.app.flags.DEFINE_float("lon_min", -71.0,
                          "ROI")
tf.app.flags.DEFINE_float("lon_max", -68.0,
                          "ROI")
tf.app.flags.DEFINE_float("onehot_lat_reso", 0.01,
                          "Resolution of the lat one-hot vector (degree)")
tf.app.flags.DEFINE_float("onehot_lon_reso",  0.01,
                          "Resolution of the lat one-hot vector (degree)")
tf.app.flags.DEFINE_float("onehot_sog_reso", 1,
                          "Resolution of the SOG one-hot vector (knot)")
tf.app.flags.DEFINE_float("onehot_cog_reso", 5,
                          "Resolution of the COG one-hot vector (degree)")

# A contrario detection flags.
tf.app.flags.DEFINE_float("cell_lat_reso", 0.1,
                          "Lat resolution of each small cell when applying local thresholding")
tf.app.flags.DEFINE_float("cell_lon_reso",  0.1,
                          "Lon nesolution of each small cell when applying local thresholding")

tf.app.flags.DEFINE_float("contrario_eps", 1e-9,
                          "A contrario eps.")
tf.app.flags.DEFINE_boolean("print_log", False,
                            "If true, print the current state of the program to screen.")



# Training flags.

tf.app.flags.DEFINE_boolean("normalize_by_seq_len", True,
                            "If true, normalize the loss by the number of timesteps "
                            "per sequence.")
tf.app.flags.DEFINE_float("learning_rate", 0.0003,
                          "The learning rate for ADAM.")
tf.app.flags.DEFINE_integer("max_steps", int(80000),
                            "The number of gradient update steps to train for.")
tf.app.flags.DEFINE_integer("summarize_every", 100,
                            "The number of steps between summaries.")


# Distributed training flags.
tf.app.flags.DEFINE_string("master", "",
                           "The BNS name of the TensorFlow master to use.")
tf.app.flags.DEFINE_integer("task", 0,
                            "Task id of the replica running the training.")
tf.app.flags.DEFINE_integer("ps_tasks", 0,
                            "Number of tasks in the ps job. If 0 no ps job is used.")
tf.app.flags.DEFINE_boolean("stagger_workers", True,
                            "If true, bring one worker online every 1000 steps.")

# Fix tf >=1.8.0 flags bug
tf.app.flags.DEFINE_string('f', '', 'kernel')
tf.app.flags.DEFINE_integer("data_dim", 0, "Data dimension")
tf.app.flags.DEFINE_string('log_filename', '', 'Log filename')
tf.app.flags.DEFINE_string('logdir_name', '', 'Log dir name')
tf.app.flags.DEFINE_string('logdir', '', 'Log directory')
tf.app.flags.DEFINE_string('trainingset_path', '', 'Training set path')
tf.app.flags.DEFINE_string('testset_path', '', 'Test set path')
tf.app.flags.DEFINE_integer("onehot_lat_bins", 0,
                          "Number of equal-width bins of the lat one-hot vector (degree)")
tf.app.flags.DEFINE_integer("onehot_lon_bins",  0,
                          "Number of equal-width bins the lat one-hot vector (degree)")
tf.app.flags.DEFINE_integer("onehot_sog_bins", 1,
                          "Number of equal-width bins the SOG one-hot vector (knot)")
tf.app.flags.DEFINE_integer("onehot_cog_bins", 5,
                          "Number of equal-width bins of the COG one-hot vector (degree)")
tf.app.flags.DEFINE_integer("n_lat_cells", 0,
                          "Number of lat cells")
tf.app.flags.DEFINE_integer("n_lon_cells",  0,
                          "Number of lon cells")


FLAGS = tf.app.flags.FLAGS
config = FLAGS


## CONFIGS
#===============================================

# Load encoding metadata BEFORE deriving bin counts or building the graph.
# Works with metadata.json emitted by dataset_preprocessing.py.
def _encoding_metadata(path):
    with open(path, 'r') as stream:
        info = json.load(stream)
    if info.get('normalized') is not True:
        raise ValueError('%s is not a normalized, processed dataset. Run dataset_preprocessing.py first.' % path)
    roi = info['roi']
    res = info['resolutions']
    if len(res) != 4:
        raise ValueError('Expected four resolutions in %s' % path)
    values = dict(zip(('onehot_lat_reso','onehot_lon_reso','onehot_sog_reso','onehot_cog_reso'), res))
    values.update(lat_min=roi['south'], lat_max=roi['north'], lon_min=roi['west'], lon_max=roi['east'])
    values = {key: float(value) for key, value in values.items()}
    spans = [values['lat_max']-values['lat_min'], values['lon_max']-values['lon_min'], SPEED_MAX, 360.]
    if not all(math.isfinite(x) for x in values.values()) or any(x <= 0 for x in spans+list(res)):
        raise ValueError('Invalid ROI/resolutions in %s' % path)
    bins = [math.ceil(span/float(r)) for span,r in zip(spans,res)]
    if list(info.get('bins',bins)) != bins or info.get('data_dim',sum(bins)) != sum(bins):
        raise ValueError('Inconsistent bin counts in %s' % path)
    return values, bins

_training_folder = os.path.dirname(os.path.join(config.dataset_dir, config.trainingset_name))
_metadata_path = os.path.join(_training_folder, 'metadata.json')
if os.path.isfile(_metadata_path):
    _values, _bins = _encoding_metadata(_metadata_path)
    _explicit = {arg.split('=',1)[0][2:] for arg in sys.argv[1:] if arg.startswith('--')}
    for _key, _value in _values.items():
        if _key in _explicit and not math.isclose(float(getattr(config,_key)), _value, rel_tol=0, abs_tol=1e-12):
            raise ValueError('--%s conflicts with %s. Remove the override or use a matching processed dataset.' % (_key,_metadata_path))
        setattr(config, _key, _value)
    # Validation/evaluation data must use the same geographical encoding.
    if config.testset_name:
        _test_meta = os.path.join(os.path.dirname(os.path.join(config.dataset_dir,config.testset_name)), 'metadata.json')
        if os.path.isfile(_test_meta) and os.path.abspath(_test_meta) != os.path.abspath(_metadata_path):
            _test_values, _ = _encoding_metadata(_test_meta)
            if _test_values != _values:
                raise ValueError('Training and evaluation metadata use different encodings')
    _mean_path = os.path.join(_training_folder, 'mean.pkl')
    with open(_mean_path, 'rb') as _stream:
        _mean = np.asarray(pickle.load(_stream))
    if _mean.shape != (sum(_bins),) or not np.isfinite(_mean).all():
        raise ValueError('mean.pkl must contain %d finite values for this dataset' % sum(_bins))
    print('Encoding loaded from %s; bins=%s; dimension=%d' % (_metadata_path,_bins,sum(_bins)))
else:
    print('No processed metadata.json found; using explicit/default ROI and resolutions.')

## FOUR-HOT VECTOR 
config.onehot_lat_bins = math.ceil((config.lat_max-config.lat_min)/config.onehot_lat_reso)
config.onehot_lon_bins = math.ceil((config.lon_max-config.lon_min)/config.onehot_lon_reso)
config.onehot_sog_bins = math.ceil(SPEED_MAX/config.onehot_sog_reso)
config.onehot_cog_bins = math.ceil(360/config.onehot_cog_reso)

config.data_dim  = config.onehot_lat_bins + config.onehot_lon_bins\
                 + config.onehot_sog_bins + config.onehot_cog_bins # error with data_dimension

## LOCAL THRESHOLDING
config.n_lat_cells = math.ceil((config.lat_max-config.lat_min)/config.cell_lat_reso)
config.n_lon_cells = math.ceil((config.lon_max-config.lon_min)/config.cell_lon_reso)


## PATH
if config.mode == "train":
    config.testset_name = config.trainingset_name
elif config.testset_name == "":
    config.testset_name = config.trainingset_name.replace("_train","_test")
config.trainingset_path = os.path.join(config.dataset_dir,config.trainingset_name)
config.testset_path = os.path.join(config.dataset_dir,config.testset_name)

print("Training set: " + config.trainingset_path)
print("Test set: " + config.testset_path)


# log
log_dir = config.bound + "-"\
     + os.path.basename(config.trainingset_name)\
     + "-data_dim-" + str(config.data_dim)\
     + "-latent_size-" + str(config.latent_size)\
     + "-batch_size-" + str(config.batch_size)
config.logdir = os.path.join(config.log_dir,log_dir)
if not os.path.exists(config.logdir):
    if config.mode == "train":
        os.makedirs(config.logdir)
    else:
        raise ValueError(config.logdir + " doesnt exist")

if config.log_filename == "":
    config.log_filename = os.path.basename(config.logdir)
