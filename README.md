# Deep Learning for Collective Variables (DeepCV)

DeepCV implements an unsupervised machine learning so-called DAENN for learning molecules and computing collective variables.

## Main Features

1. Data sampling
   - Selected structural properties: internal coordinate
   - SPRINT and Extended SPRINT coordinates
   - PCA components (noise reduction)
2. Dimensionality reduction
   - Linear: PCA, MDS
   - Autoencoders
     - Stacked autoencoder
     - One-Hot autoencoder
3. Generative model for generating data
   - Generative adversarial networks (GANs)
   - Variational autoencoder (future work)

## Quick installation

- Python codes
  ```sh
  cd deepcv/
  pip install -r requirements.txt
  pip install .
  ```
- C++ codes

  ```sh
  g++ -c -I/path/to/plumed/src/ -o deepcv.o deepcv.cpp
  g++ -shared -fPIC -o deepcv.so deepcv.o
  ```

  or you can use `make`

  ```sh
  make
  ```

## Usage

The following is an example command for training model of collective variables for Diels-Alder reaction using reactant trajectory's descriptors. You can call DeepCV's DAENN via either `main.py` API or `deepcv_daenn` command (register entrypoint).

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

To install all dependencies packages of DeepCV, you can follow either following ways:

1. All at once (for the users)

   - Using PIP
     - `pip install --upgrade pip`
     - `pip install -r requirements.txt`
   - Using Conda (recommended)
     - `conda update --all -y`
     - `conda install --file requirements.txt`

2. Install packages separately (recommended for the developers)

   - NumPy >= 1.22.2
     - `pip install --upgrade numpy==1.22.2`
   - TensorFlow + Keras 2.8.0 (CPU+GPU)
     - `pip install tensorflow`
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

1. Time-series data with autoencoder
2. Improve neural network algorithm for large systems
3. Implement algorithm into MD packages and test the performance

## Done

1. Dense neural nets with backpropagation
   - Single and multi-input simple nets
   - Single and multi-input stacked autoencoder
2. Applied techniques to overcome overfit
   - Mini-Batch training and normalization
   - Drop-out
3. Generate PLUMED input file
4. Molecular-based loss function
5. Model training
   - Avoid saturation
   - Epoch iteration
   - Penalty update
   - GPU acceleration
6. CV coordinate space propagation using loss maximization technique
7. Analysis tools
   - Feature importance

---

## Authors

1. Rangsiman Ketkaew (rangsiman.ketkaew@chem.uzh.ch)
2. Sandra Luber (sandra.luber@chem.uzh.ch)
