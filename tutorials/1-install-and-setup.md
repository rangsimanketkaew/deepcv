# Install DeepCV on your local machine <!-- omit in toc -->

- [Step 1: Clone DeepCV repo or download a tarball to your local machine](#step-1-clone-deepcv-repo-or-download-a-tarball-to-your-local-machine)
- [Step 2: Environment setup & install dependencies](#step-2-environment-setup--install-dependencies)
- [Step 3: Check if GPU is available for TF](#step-3-check-if-gpu-is-available-for-tf)

## Step 1: Clone DeepCV repo or download a tarball to your local machine

For git, clone with SSH is recommended:
```sh
git clone git@gitlab.uzh.ch:lubergroup/deepcv.git
```

## Step 2: Environment setup & install dependencies

```sh
cd deepcv/
conda create -n deepcv python==3.8
conda activate deepcv
conda update --all -y
pip install -r requirements.txt # or conda install --file requirements.txt
python setup.py install
```

Optional: for C++ API
```sh
g++ -c -I/path/to/plumed/src/ -o deepcv.o deepcv.cpp
g++ -shared -fPIC -o deepcv.so deepcv.o
```
or just type
```sh
make
```

## Step 3: Check if GPU is available for TF

Execute this script to check if TensorFlow >= 2.x can see GPU.
```python
import tensorflow as tf

assert tf.test.is_gpu_available(), "TF can't see GPU on this machine."
assert tf.test.is_built_with_cuda(), "TF was not built with CUDA."
```
