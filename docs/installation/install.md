# Install DeepCV

## Install dependencies

DeepCV requires the following libraries (the supported version for each library is as of the current version of DeepCV):

```
tensorflow-gpu==2.10.0
numpy
scikit-learn
scipy
matplotlib
ase
imageio
natsort
tabulate
tqdm
pydot
```

One can use this command to install all of them
```sh
pip install -r requirements.txt # or conda install --file requirements.txt
```

Note that installing tensorflow may be sometimes tricky, depending on system environment. 
Please consult its official website in case you have errors.

## Install DeepCV (Python)

```sh
cd deepcv/
python3 -m pip install .
# Test calling the main API
main.py # or deepcv
```

## Installing DeepCV (C++)

Standalone shared library file
```sh
export LIB_TF=/usr/local/tensorflow/

g++ -Wall -fPIC -o deepcv.o deepcv.cpp \
    -O3 -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++14 -fPIC \
    -I${LIB_TF}/include/ -L${LIB_TF}/lib \
    -ltensorflow -ltensorflow_cc -ltensorflow_framework
```
or just type
```sh
make CXXFLAGS="-std=c++14 -fPIC"
```
and then build an object file
```sh
g++ -shared -o deepcv.so deepcv.o
```

## Check if GPU is available for TF

Execute this script to check if TensorFlow 2.x can see GPU.
```python
import tensorflow as tf

assert tf.test.is_gpu_available(), "TF can't see GPU on this machine."
assert tf.test.is_built_with_cuda(), "TF was not built with CUDA."
```
