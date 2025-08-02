# Overview of How to Install DeepCV

## Dependencies

DeepCV requires the following libraries (the supported version for each library is as of the current version of DeepCV):

```
tensorflow
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

## Clone DeepCV repo or download a tarball to your local machine

The source code of DeepCV is available free of charge for academic purposes on our [GitLab](https://gitlab.uzh.ch/lubergroup/deepcv).

- For git, clone with SSH is recommended:
    ```sh
    git clone git@gitlab.uzh.ch:lubergroup/deepcv.git
    ```

- For tarball:
    ```sh
    wget https://gitlab.uzh.ch/lubergroup/deepcv/-/archive/master/deepcv-master.tar
    tar -xvf deepcv-master.tar
    mv deepcv-master deepcv
    ```

## Install with package manager

The easiest way to install DeepCV is to use a Python package manager; either `pip` or `conda`.
However, we suggest you create a separate working environment for DeepCV to prevent redundancy of the package dependencies and/or version conflict; and this can be easily done by `conda`.

## For developers

For those who want to contribute to the DeepCV and PLUMED using TensorFlow C++ API, we suggest you compile and install source code manually. More details are available in the [Install DeepCV C++](deepcv-cpp.md) section.
