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
  conda install --file requirements.txt
  python setup.py install
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

The following is an example command for training model of collective variables for Diels-Alder reaction using reactant trajectory's descriptors:

```sh
ae_train -i input_ae_DA.json -d distance.npz angle.npz torsion.npz
```

Note: A complete tutorial on DeepCV is available [here](tutorials/).

---

## Development

- Python 3.6 or a newer version
- Use git control: `git clone https://gitlab.uzh.ch/lubergroup/deepcv.git`
- Please write function docstring, comments for difficult-to-understand code and documentation for the module and package you are developing
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
   - TensorFlow 2.8.0
     - `pip install tensorflow-gpu`
     - `conda install tensorflow-gpu` (recommended)
   - Keras 2.8.0 (TF API)
     - `pip install keras`
   - NVIDIA GPU and CUDA 10.1 (for GPU enabled)
     - https://developer.nvidia.com/cuda-toolkit-archive
   - cuDNN v7.6.4 (September 27, 2019), for CUDA 10.1
     - https://developer.nvidia.com/rdp/cudnn-archive
   - pydot (for `keras.utils.vis_utils.plot_model`)
     - `conda install pydot`
   - other important scientific packages
     - scipy, scikit-learn, matplotlib, pandas

3. DeepCV C++ makes use of JSON parser
   - https://github.com/nlohmann/json

## In Progress

1. Time-series data with autoencoder
2. Improve neural network algorithm for large systems
3. Implement algorithm into MD packages and test the performance

## Done

1. Dense neural nets with back propagation
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
   - GPU accelerated
6. CV coordinate space propagation using loss maximization technique
7. Analysis tools
   - Feature importance

---

## Authors

1. Rangsiman Ketkaew (rangsiman.ketkaew@chem.uzh.ch)
2. Sandra Luber (sandra.luber@chem.uzh.ch)
