# Deep Learning for Collective Variables (DeepCV)

DeepCV is a tool to learn collective variables (CVs) of a chemical process for enhanced sampling simulation, using *deep autoencoder neural network* (DAENN) model (non-iterative approach).

Website: https://lubergroup.pages.uzh.ch/deepcv/

## Main Features

1. Molecular descriptors
   - Internal coordinates
   - SPRINT and eXtended SPRINT coordinates
2. Stacked autoencoder
3. Loss customization with minimaxation technique to learn CVs in expanded configurational space
   - Primary loss: main loss with regularization
   - Secondary loss: Additional loss to be maximized to expand the CV space
4. Generative model for generating data
   - Generative adversarial networks (GANs)
5. Can interface with [PLUMED](https://www.plumed.org/)
6. Tools
   - Input file generator (GUI) for PLUMED and CP2K
   - Feature importance analysis
   - Sampling convergence assessment

## Quick installation

- Install with `pip`:
  ```sh
  cd deepcv
  pip install -r requirements.txt
  pip install .
  ```

- Install with `conda`:
  ```sh
  conda env create --file environment.yml
  conda activate deepcv
  ```

<details>
<summary>Dependencies</summary>

Install dependencies (packages required by DeepCV) separately (recommended for the developers)

  - All dependencies are listed in [requirements.txt](./requirements.txt)
  - DeepCV C++ makes use of JSON parser: https://github.com/nlohmann/json
</details>

## Usage

You can call DeepCV's module via registered entry point `deepcv` followed by a module, e.g.:

```sh
deepcv daenn -i input_ae_DA.json
```

or by executing `main.py` API script:

```sh
python deepcv/src/main.py daenn -i input_ae_DA.json
```

where `input_ae_DA.json` is [an input file of DAENN for Diels-Alder reaction](input/input_ae_DA.json).

## Development

- Python 3.9 - 3.12
- TensorFlow 2.16 and Keras 3 or newer version
- Use git control: `git clone https://gitlab.uzh.ch/lubergroup/deepcv.git`
- Please write function docstring and comment for difficult-to-understand code
- Document modules and packages you are developing
- Format codes with [Black](https://github.com/psf/black)
- Send pull-request to master with an explanation, for example, what you contribute, how it works, and usefulness

## In Progress

1. Variational autoencoder
2. Improve neural network algorithm for large systems e.g. metal-oxide surface
3. Improve code compatibility between TensorFlow, PLUMED, and CP2K

---

## Authors

1. Rangsiman Ketkaew (rangsiman.ketkaew@chem.uzh.ch)
2. Sandra Luber (sandra.luber@chem.uzh.ch)
