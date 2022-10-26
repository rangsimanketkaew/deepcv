# Deep Learning for Collective Variables (DeepCV)

DeepCV is a computer code that implements our developed unsupervised machine learning called *deep autoencoder neural network* (DAENN) for learning low-dimensional collective variables (CVs) aka slow-mode reaction coordinates of a set of molecules for enhanced sampling.

Website: https://lubergroup.pages.uzh.ch/deepcv/

## Main Features

1. Molecular features
   - Internal coordinates
   - SPRINT and eXtended SPRINT coordinates
2. Dense autoencoder neural nets
   - Single and multi-input simple and stacked autoencoder
   - Avoid saturation
   - GPU acceleration
3. Learn CVs in expanded configurational space
   - Customized loss functions with minimaxation technique
     - Primary loss: main loss with regularization
     - Secondary loss: Additional loss to be maximized to expand the CV space
   - Self-directed expansion of configurational space
4. Generative model for generating data
   - Generative adversarial networks (GANs)
   - Variational autoencoder (future work)
5. Can interface with PLUMED and CP2K
6. Input file generator (GUI) for PLUMED and CP2K
7. Analysis tools
   - Feature importance
   - Sampling convergence assessment

## Quick installation

- Python codes
  ```sh
  cd deepcv/
  pip install -r requirements.txt
  pip install .
  ```
- C++ codes

  ```sh
  cd deepcv/
  export LIB_TF=/usr/local/tensorflow/

  g++ -Wall -fPIC -o deepcv.o deepcv.cpp \
      -O3 -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++14 -fPIC \
      -I${LIB_TF}/include/ -L${LIB_TF}/lib \
      -ltensorflow -ltensorflow_cc -ltensorflow_framework

  g++ -shared -fPIC -o deepcv.so deepcv.o
  ```
  or you can use `make`
  ```sh
  make CXXFLAGS="-std=c++14 -fPIC"
  ```

## Usage

The following is an example command for training model of CVs for Diels-Alder reaction using features extracted from reactant trajectory. You can call DeepCV's DAENN via either `main.py` API or `deepcv_daenn` command (register entrypoint).

```sh
python main.py daenn -i input_ae_DA.json
```

or 

```sh
deepcv_daenn -i input_ae_DA.json
```

---

## Development

- Python 3.6 or a newer version
- Use git control: `git clone https://gitlab.uzh.ch/lubergroup/deepcv.git`
- Please write function docstring and comment for difficult-to-understand code
- Document modules and packages you are developing
- Format codes with [Black](https://github.com/psf/black)
- Send pull-request to master with an explanation, for example, what you contribute, how it works, and usefulness

## Packages requirements

To install all dependencies packages of DeepCV, you can follow either following way:

1. All at once (for the users)

   - Using PIP
     - `pip install --upgrade pip`
     - `pip install -r requirements.txt`
   - Using Conda (recommended)
     - `conda update --all -y`
     - `conda install --file requirements.txt`

2. Install packages separately (recommended for the developers)

   - NumPy >= 1.22.2
     - E.g., `pip install --upgrade numpy==1.23.4`
   - TensorFlow + Keras 2.10.0
     - `pip install tensorflow` (CPU)
     - `conda install tensorflow-gpu` (+ CUDA & cuDNN)
   - NVIDIA GPU and CUDA 10.1 (for GPU enabled)
     - https://developer.nvidia.com/cuda-toolkit-archive
   - cuDNN v7.6.4 (September 27, 2019), for CUDA 10.1
     - https://developer.nvidia.com/rdp/cudnn-archive
   - pydot (for `keras.utils.vis_utils.plot_model`)
     - `conda install pydot`
   - other important packages are listed in [requirements.txt](./requirements.txt)

3. DeepCV C++ makes use of JSON parser
   - https://github.com/nlohmann/json

## In Progress

1. Variational autoencoder
2. Improve neural network algorithm for large systems e.g. metal-oxide surface
3. Improve code compatibility between TensorFlow, PLUMED, and CP2K

---

## Authors

1. Rangsiman Ketkaew (rangsiman.ketkaew@chem.uzh.ch)
2. Sandra Luber (sandra.luber@chem.uzh.ch)
