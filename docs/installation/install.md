# Installing DeepCV

## Installing dependencies

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
