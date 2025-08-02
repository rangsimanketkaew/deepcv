# Prepare DeepCV Input File

DeepCV's input file is written in a JSON file format (dictionary-like).
Input examples provided in [deepcv/inputs/](https://gitlab.uzh.ch/lubergroup/deepcv/-/tree/master/input) folder 
show the configurations for training a DAENN model with five hidden layers (fixed symmetric architecture):

- The first three hidden layers contain two encoded layers and one latent encoded layer (middle layer)
- The last two layers are two decoded layers for reconstruction

Here is the example input file for training DAENN CV of Diels-Alder reaction: [input_ae_DA.json](https://gitlab.uzh.ch/lubergroup/deepcv/-/blob/master/input/input_ae_DA.json)

## Keys

#### Project
| Key              | Definition             | Value          |
|------------------|------------------------|----------------|
| `name`           | Project name           | String         |
| `neural_network` | Type of neural network | `daenn`, `gan` |

#### Dataset
| Key               | Definition                                          | Value                    |
|-------------------|-----------------------------------------------------|--------------------------|
| `primary`         | A list of dataset files for primary loss function   | String                   |
| `secondary`       | A list of dataset files for secondary loss function | String                   |
| `split`           | Split dataset                                       | Logical: `true`, `false` |
| `split_ratio`     | Ratio for splitting (for training set)              | Integer: `0.8`           |
| `shuffle`         | Shuffle the data points                             | Logical: `true`, `false` |
| `normalize_scale` | Normalization scaling value                         | Float: `0.0`             |
| `max_scale`       | Maximum scaling value                               | Integer: `1`             |

#### Model
| Key                  | Definition                     | Value                                                                      |
|----------------------|--------------------------------|----------------------------------------------------------------------------|
| `optimizer`          | Optimizer                      | `Adadelta`, `Adagrad`, `Adam`, `Adamax`, `Ftrl`, `Nadam`, `RMSprop`, `SGD` |
| `main_loss`          | Primary loss function          | `MSE`, `MAE`                                                               |
| `penalty_loss`       | Secondary loss function        | `MSE`, `MAE`                                                               |
| `loss_weights`       | A list of weight for each loss | `[1, 0.1]`                                                                 |
| `num_epoch`          | Number of training epochs      | Integer: `1000`                                                            |
| `batch_size`         | Batch size                     | Integer: `55`                                                              |
| `save_every_n_epoch` | Save model every N-th epoch    | Integer: `10`                                                              |

#### Neural network
| Key         | Definition                                          | Value                     |
|-------------|-----------------------------------------------------|---------------------------|
| `units`     | A list of number of neurons per hidden layer        | Integer                   |
| `act_funcs` | A list of activation function for each hidden layer | `relu`, `sigmoid`, `tanh` |

#### Performance
| Key          | Definition     | Value                    |
|--------------|----------------|--------------------------|
| `enable_gpu` | Train on GPU   | Logical: `true`, `false` |
| `gpus`       | Number of GPUs | Integer                  |

#### Settings
| Key                | Definition                 | Value                    |
|--------------------|----------------------------|--------------------------|
| `verbosity`        | Level of output printing   | Integer: `1`             |
| `show_summary`     | Show DAENN summary         | Logical: `true`, `false` |
| `save_model`       | Save trained model         | Logical: `true`, `false` |
| `save_weights`     | Save weights               | Logical: `true`, `false` |
| `save_weights_npz` | Save weights in npz format | Logical: `true`, `false` |
| `save_biases_npz`  | Save biases in npz format  | Logical: `true`, `false` |
| `save_graph`       | Save TensorFlow graph      | Logical: `true`, `false` |
| `save_loss`        | Save loss                  | Logical: `true`, `false` |
| `show_loss`        | Show loss                  | Logical: `true`, `false` |
| `save_metrics`     | Save metrics               | Logical: `true`, `false` |
| `show_metrics`     | Show metrics               | Logical: `true`, `false` |

#### Output
| Key               | Definition                           | Value  |
|-------------------|--------------------------------------|--------|
| `out_dir`         | Path for output directory            | String |
| `out_tb`          | Name of output TensorBoard           | String |
| `out_model`       | Name of output model                 | String |
| `out_weights`     | Name of output weights               | String |
| `out_weights_npz` | Name of output weights in npz format | String |
| `out_biases_npz`  | Name of output biases in npz format  | String |
| `loss_plot`       | Name of loss plot                    | String |
| `metrics_plot`    | Name of metrics plot                 | String |
