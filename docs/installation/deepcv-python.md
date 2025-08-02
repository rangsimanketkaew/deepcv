# Install DeepCV Python

## Install with pip

It is recommended to use `pip` to install DeepCV

```sh
pip install -r requirements.txt 
pip install .
```

Test calling `deepcv` with

```sh
deepcv
# or
deepcv calc_rep
# or 
deepcv daenn
```

> Installing `tensorflow` can be tricky, depending on operating system and system environment.
> Please check [Install TensorFlow 2](https://www.tensorflow.org/install) for more details.

## Install with conda

Setup environment & install dependencies

```sh
conda env create --file environment.yml
conda activate deepcv
```

This command will create a conda environment named `deepcv` based on parameters defined in `environment.yml` file. 
You can also change the name of environment using argument `--name NEW_NAME`.

If you have already an empty environment, you can install the dependencies with

```sh
conda install --file requirements.txt
```

## Check if GPU is available for TF

Execute this script to check if TensorFlow 2.x can see GPU.

```python
import tensorflow as tf

assert tf.test.is_gpu_available(), "TF can't see GPU on this machine."
assert tf.test.is_built_with_cuda(), "TF was not built with CUDA."
```
