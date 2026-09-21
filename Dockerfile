# CPU environment for CIA-Oceanix/GeoTrackNet (TensorFlow 1.12).
# Build: docker build --platform linux/amd64 -t geotracknet .
# Run from the repository: docker run --rm -it --platform linux/amd64 \
#   -v "$PWD:/workspace" -w /workspace geotracknet
# The original Conda requirements.yml is intentionally not installed.
FROM python:3.6.15-slim-buster

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

# Keep packaging tools compatible with Python 3.6.
RUN python -m pip install --no-cache-dir \
    pip==21.3.1 setuptools==59.6.0 wheel==0.37.1

# Preserve the model's TF1/Sonnet1 APIs and the main original package versions.
# Pip's tensorflow package is CPU-only at version 1.12; CUDA is not required.
RUN python -m pip install --no-cache-dir \
    tensorflow==1.12.0 \
    tensorflow-probability==0.5.0 \
    dm-sonnet==1.27 \
    numpy==1.15.4 \
    scipy==1.4.1 \
    pandas==1.0.2 \
    matplotlib==3.1.3 \
    scikit-learn==0.22.1 \
    pyproj==2.4.1 \
    tqdm==4.43.0 \
    h5py==2.10.0 \
    protobuf==3.11.3 \
    tensorboard==1.12.2 \
    grpcio==1.27.2 \
    absl-py==0.9.0 \
    gast==0.2.2 \
    astor==0.8.0 \
    keras-applications==1.0.8 \
    keras-preprocessing==1.1.0 \
    six==1.15.0 \
    semantic-version==2.6.0 \
    contextlib2==0.6.0.post1 \
    Werkzeug==1.0.0 \
    Markdown==3.1.1 \
    termcolor==1.1.0 \
    wrapt==1.12.1 \
    && python -m pip check

# Fail at build time on import, native-library, or basic graph incompatibility.
RUN python -c "import tensorflow as tf; import tensorflow_probability as tfp; import sonnet as snt; import numpy, scipy, pandas, matplotlib, sklearn, pyproj, h5py; assert tf.__version__ == '1.12.0'; x = snt.Linear(2)(tf.ones([1, 3])); d = tfp.distributions.Normal(0., 1.); sess = tf.Session(); sess.run(tf.global_variables_initializer()); print('Sonnet:', sess.run(x), 'TFP:', sess.run(d.log_prob(0.))); sess.close()"

WORKDIR /workspace
CMD ["bash"]
