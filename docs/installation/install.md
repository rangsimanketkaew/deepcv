# Installing DeepCV

## Installing dependencies

DeepCV requires the following libraries (the supported version for each library is as of the current version of DeepCV):

```
numpy==1.22.2
tensorflow==2.8.0
tensorflow-addons
scikit_learn==1.0.2
scipy==1.8.0
sympy==1.9
matplotlib
ase
imageio
natsort
tabulate
bayesian-optimization
tqdm
pydot
graphviz
```

One can use this command to install all of them
```sh
pip install -r requirements.txt # or conda install --file requirements.txt
```

## Installing DeepCV

```sh
cd deepcv/
conda create -n deepcv python==3.8
conda activate deepcv
conda update --all -y
pip install -r requirements.txt # or conda install --file requirements.txt
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

## Check if GPU is available for TF

Execute this script to check if TensorFlow 2.x can see GPU.
```python
import tensorflow as tf

assert tf.test.is_gpu_available(), "TF can't see GPU on this machine."
assert tf.test.is_built_with_cuda(), "TF was not built with CUDA."
```
