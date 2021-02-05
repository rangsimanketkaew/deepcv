# Deep learning-based collective variables (DeepCV)

Unsupervised learning for discovering hidden collective variables on free energy surface.

## Usage
The following is an example command for training model of collective variables for Diels-Alder reaction using reactant trajectory's descriptors:
```sh
cd deepcv/
python -m src.modules.ae_train \
    -i input_ae_DA.json \
    -d DNN_DA_R_distance.npz DNN_DA_R_angle.npz DNN_DA_R_torsion.npz DNN_DA_R_xSPRINT.npz \
    -k dist angle torsion xsprint
```

Note: A complete tutorial on DeepCV is available [here](tutorials/).

---

## Development
- Python 3.6 or a newer version
- Use git control: `git clone https://gitlab.uzh.ch/lubergroup/deepcv.git`
- Please write function docstring, comments for difficult-to-understand code and documentation for the module and package you are developing
- Send pull-request to master with an explanation for example what you contribute, how it works, and usefulness

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
    - NumPy >= 1.19.1
      - `pip install --upgrade numpy==1.19.1`
    - TensorFlow >= 2.2.0
      - `pip install tensorflow-gpu`
      - `conda install tensorflow-gpu`  (recommended)
    - Keras 2.4.0 (TF API)
      - `pip install keras`
    - NVIDIA GPU and CUDA 10.1 (for GPU enabled)
      - https://developer.nvidia.com/cuda-toolkit-archive
    - cuDNN v7.6.4 (September 27, 2019), for CUDA 10.1
      - https://developer.nvidia.com/rdp/cudnn-archive
    - pydot (for `keras.utils.vis_utils.plot_model`)
      - `conda install pydot`
    - other important scientific packages
      - scipy, scikit-learn, matplotlib, pandas

## Roadmap
1. Data sampling
    - Selected structural properties
    - SPRINT coordinates
    - PCA components (noise reduction)
2. Dimensionality reduction
    - Linear: PCA, MDS
    - Autoencoders
      - Stacked autoencoder
      - One-Hot autoencoder
3. Generative model for generating data
    - Generative adversarial networks (GANs) (future work)
    - Variational autoencoder (future work)

## To Do
1. Development of loss function for autoencoder
2. Development of regularizer
3. Time-series data with autoencoder

## In Progress
1. Analysis of CV coordinate space

## Done
1. Dense neural nets with back propagation
    - Single and multi-input simple nets
    - Single and multi-input stacked autoencoder
2. Applied techniques to overcome overfit
    - Mini-Batch training and normalization
    - Drop-out
3. Generate PLUMED input file
4. Model training
    - Avoid saturation
    - Epoch iteration
    - Penalty update
    - GPU accelerated

---

## Author
1. Rangsiman Ketkaew (rangsiman.ketkaew@chem.uzh.ch)
