"""
Deep Learning for Collective Variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
28/11/2020 : v1.0 [RK]
02/08/2025 : v2.0 supports TensorFlow 2.16 and Keras 3 [RK]
"""

"""
Deep Autoencoder Neural Network (DAENN) for learning collective variables from molecular representations
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for relative imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import argparse
import logging
from utils import util
from datetime import datetime
from inspect import getmembers, isfunction
from pathlib import Path
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import List, Optional

import functools
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, LeakyReLU, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import Callback, TensorBoard
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from modules import loss
from tools import ae_visual

logging = logging.getLogger("DeepCV")


DEFAULT_SAVE_INTERVAL = 10
DEFAULT_DPI = 192
DEFAULT_GAMMA = 0.8
FLOAT_PRECISION = np.float32
RANDOM_STATE = 42


@dataclass
class ProjectConfig:
    """Configuration for project settings."""

    name: str
    neural_network: str

    @classmethod
    def from_json(cls, json_data):
        project_data = json_data.get("project", {})
        return cls(
            name=project_data.get("name", "Unknown"),
            neural_network=project_data.get("neural_network", "daenn"),
        )


@dataclass
class DatasetConfig:
    """Configuration for dataset settings."""

    primary: List[str]
    secondary: List[str]
    split: bool
    split_ratio: float
    shuffle: bool
    normalize_scale: float
    max_scale: float
    keys: Optional[List[str]] = None

    @classmethod
    def from_json(cls, json_data):
        dataset_data = json_data.get("dataset", {})
        return cls(
            primary=dataset_data.get("primary", []),
            secondary=dataset_data.get("secondary", []),
            split=dataset_data.get("split", True),
            split_ratio=dataset_data.get("split_ratio", 0.8),
            shuffle=dataset_data.get("shuffle", True),
            normalize_scale=dataset_data.get("normalize_scale", 0.0),
            max_scale=dataset_data.get("max_scale", 1.0),
            keys=dataset_data.get("keys", None),
        )

    def validate(self):
        """Validate dataset configuration."""
        if not self.primary:
            raise ValueError("Primary dataset list cannot be empty")
        if not 0.1 <= self.split_ratio <= 0.9:
            raise ValueError("Split ratio must be between 0.1 and 0.9")
        if self.keys and len(self.keys) != len(self.primary + self.secondary):
            raise ValueError("Number of keys must match total number of datasets")


@dataclass
class ModelConfig:
    """Configuration for model settings."""

    optimizer: str
    main_loss: str
    penalty_loss: str
    loss_weights: List[float]
    num_epoch: int
    batch_size: int
    save_every_n_epoch: Optional[int] = None

    @classmethod
    def from_json(cls, json_data):
        model_data = json_data.get("model", {})
        num_epoch = model_data.get("num_epoch", 100)
        save_every_n_epoch = model_data.get("save_every_n_epoch", None)

        # Calculate default save_every_n_epoch if not provided
        if save_every_n_epoch is None:
            save_every_n_epoch = max(
                int(num_epoch / DEFAULT_SAVE_INTERVAL), DEFAULT_SAVE_INTERVAL
            )

        return cls(
            optimizer=model_data.get("optimizer", "adam"),
            main_loss=model_data.get("main_loss", "mse"),
            penalty_loss=model_data.get("penalty_loss", "mse"),
            loss_weights=model_data.get("loss_weights", [1.0, 1.0]),
            num_epoch=num_epoch,
            batch_size=model_data.get("batch_size", 32),
            save_every_n_epoch=int(save_every_n_epoch),
        )

    def validate(self):
        """Validate model configuration."""
        if self.num_epoch <= 0:
            raise ValueError("Number of epochs must be positive")
        if self.batch_size < 0:
            raise ValueError("Batch size must be non-negative (0 for full batch)")
        if len(self.loss_weights) != 2:
            raise ValueError("Loss weights must have exactly 2 values")


@dataclass
class NetworkConfig:
    """Configuration for network architecture."""

    units: List[int]
    act_funcs: List[str]

    @classmethod
    def from_json(cls, json_data):
        network_data = json_data.get("network", {})
        return cls(
            units=network_data.get("units", [256, 128, 64, 128, 256]),
            act_funcs=network_data.get(
                "act_funcs", ["relu", "relu", "relu", "relu", "relu"]
            ),
        )

    def validate(self):
        """Validate network configuration."""
        if len(self.units) != 5:
            raise ValueError("Network must have exactly 5 layers (units list)")
        if len(self.act_funcs) != 5:
            raise ValueError("Network must have exactly 5 activation functions")
        if any(u <= 0 for u in self.units):
            raise ValueError("All units must be positive integers")


@dataclass
class PerformanceConfig:
    """Configuration for performance settings."""

    enable_gpu: bool

    @classmethod
    def from_json(cls, json_data):
        perf_data = json_data.get("performance", {})
        return cls(enable_gpu=perf_data.get("enable_gpu", True))


@dataclass
class SettingsConfig:
    """Configuration for general settings."""

    verbosity: int
    show_summary: bool
    save_model: bool
    save_weights: bool
    save_weights_npz: bool
    save_biases_npz: bool
    save_graph: bool
    save_loss: bool
    show_loss: bool
    save_metrics: bool
    show_metrics: bool

    @classmethod
    def from_json(cls, json_data):
        settings_data = json_data.get("settings", {})
        return cls(
            verbosity=settings_data.get("verbosity", 1),
            show_summary=settings_data.get("show_summary", True),
            save_model=settings_data.get("save_model", True),
            save_weights=settings_data.get("save_weights", False),
            save_weights_npz=settings_data.get("save_weights_npz", False),
            save_biases_npz=settings_data.get("save_biases_npz", False),
            save_graph=settings_data.get("save_graph", True),
            save_loss=settings_data.get("save_loss", True),
            show_loss=settings_data.get("show_loss", False),
            save_metrics=settings_data.get("save_metrics", True),
            show_metrics=settings_data.get("show_metrics", False),
        )


@dataclass
class OutputConfig:
    """Configuration for output settings."""

    out_dir: str
    out_tb: str
    out_model: str
    out_weights: str
    out_weights_npz: str
    out_biases_npz: str
    loss_plot: str
    metrics_plot: str

    @classmethod
    def from_json(cls, json_data):
        output_data = json_data.get("output", {})
        return cls(
            out_dir=output_data.get("out_dir", "./output"),
            out_tb=output_data.get("out_tb", "tensorboard"),
            out_model=output_data.get("out_model", "model"),
            out_weights=output_data.get("out_weights", "weights.h5"),
            out_weights_npz=output_data.get("out_weights_npz", "weights.npz"),
            out_biases_npz=output_data.get("out_biases_npz", "biases.npz"),
            loss_plot=output_data.get("loss_plot", "loss_history.png"),
            metrics_plot=output_data.get("metrics_plot", "metrics_history.png"),
        )


@dataclass
class DAENNConfig:
    """Complete configuration for DAENN training."""

    project: ProjectConfig
    dataset: DatasetConfig
    model: ModelConfig
    network: NetworkConfig
    performance: PerformanceConfig
    settings: SettingsConfig
    output: OutputConfig

    @classmethod
    def from_json(cls, json_data):
        """Create configuration from JSON data with validation."""
        config = cls(
            project=ProjectConfig.from_json(json_data),
            dataset=DatasetConfig.from_json(json_data),
            model=ModelConfig.from_json(json_data),
            network=NetworkConfig.from_json(json_data),
            performance=PerformanceConfig.from_json(json_data),
            settings=SettingsConfig.from_json(json_data),
            output=OutputConfig.from_json(json_data),
        )

        # Validate all configurations
        config.validate()
        return config

    def validate(self):
        """Validate all configuration sections."""
        try:
            self.dataset.validate()
            self.model.validate()
            self.network.validate()
            logging.info("Configuration validation passed")
        except ValueError as e:
            logging.error(f"Configuration validation failed: {e}")
            raise


class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()

    def add_dataset(
        self,
        train_set: List[np.ndarray],
        test_set: List[np.ndarray],
        num_primary: int,
        num_secondary: int,
    ) -> None:
        """Add dataset after creating an instance of Autoencoder class

        Args:
            train_set (list): List containing train sets (NumPy array).
                            The feature of all set must have the same shape.
            test_set (list): List containing test sets (NumPy array).
                            The feature of all set must have the same shape.
            num_primary (int): Number of primary datasets (arrays).
            num_secondary (int): Number of secondary datasets (arrays).
        """
        self.train_set = train_set
        self.test_set = test_set
        self.num_primary = num_primary
        self.num_secondary = num_secondary

    def generate_input(self, units, i):
        """Generate input layer for model.

        Args:
            units (int): Number of units/nodes of a layer
            i (int): Index of layer

        Returns:
            Input (obj): Tensorflow input layer
        """
        return Input(shape=(units,), dtype=tf.float32, name="input_layer_" + str(i))

    def generate_output(self, units, i):
        """Generate output layer for model.

        Args:
            units (int): Number of units/nodes of a layer
            i (int): Index of layer

        Returns:
            Input (obj): Tensorflow output layer
        """
        return Dense(units, dtype=tf.float32, name="output_layer_" + str(i))

    def build_network(self, output_name="daenn_output", **layer_params):
        """Multiple input fully-connected feedforward neural network.
        This network comprises input layer(s), 5 hidden layers, and output layer(s).

        Args:
            output_name (str, optional): Name of the output layer. Defaults to "daenn_output".

        Keyword Args:
            units (list): Number of neurons for each hidden layer
            act_funcs (list): Activation function for each hidden layer
        """
        self.units_1, self.units_2, self.units_3, self.units_4, self.units_5 = (
            layer_params["units"]
        )
        self.func_1, self.func_2, self.func_3, self.func_4, self.func_5 = layer_params[
            "act_funcs"
        ]

        # ---------
        # Create initial placeholder layer for each input dataset
        # ---------
        self.size_inp = [i.shape[1] for i in self.train_set]
        self.list_inp = [
            self.generate_input(self.size_inp[i], i + 1)
            for i in range(len(self.size_inp))
        ]

        # ---------
        # Apply custom layer
        # ---------
        # self.list_inp[-1] = layer.LayerWithRate()(self.list_inp[-1])  # last layer

        # ---------
        # Combine primary datasets
        # Input: I1 + I2 + ... = Inp
        # ---------
        # If there is only one dataset provided, then don't need to concatenate
        if len(self.list_inp) == 1 or len(self.list_inp) == 2:
            self.primary_inp = self.list_inp[0]
        # otherwise merge them (only primary layers)
        else:
            self.primary_inp = Concatenate(axis=-1)(
                self.list_inp[: self.num_primary]
            )  # Merge branches

        # merge again with the rest of placeholder layer (secondary dataset(s))
        self.secondary_inp = self.list_inp[self.num_primary :]
        self.inputs = Concatenate(axis=-1)([self.primary_inp] + self.secondary_inp)

        # ---------
        # Encoder: Inp --> H1 --> H2 --> H3
        # latent = feature representation
        # ---------
        self.encoded1 = Dense(self.units_1, activation=self.func_1, use_bias=True)(
            self.inputs
        )
        self.encoded2 = Dense(self.units_2, activation=self.func_2, use_bias=True)(
            self.encoded1
        )
        self.latent = Dense(self.units_3, activation=self.func_3, use_bias=True)(
            self.encoded2
        )

        # ---------
        # Decoder: H3 --> H4 --> H5 --> Out
        # output = reconstruction
        # ---------
        self.decoded1 = Dense(self.units_4, activation=self.func_4, use_bias=True)(
            self.latent
        )
        self.decoded2 = Dense(self.units_5, activation=self.func_5, use_bias=True)(
            self.decoded1
        )
        self.outputs = Dense(self.inputs.shape[1], activation=None, use_bias=True)(
            self.decoded2
        )

        # ---------
        # split output into sub-tensors
        # Output: Out --> O1 + O2 + ...
        # ---------
        # size of primary layer [ : a]
        if type(self.primary_inp.shape) == tuple:
            size_p = list(self.primary_inp.shape)[1]
        else:
            size_p = self.primary_inp.shape.as_list()[1]
        # size of secondary layer [a : ]
        size_s = functools.reduce(lambda x, y: x + y, self.size_inp[self.num_primary :])

        split_layer = Lambda(
            lambda x: tf.split(x, [size_p, size_s], axis=1), name=output_name
        )
        self.list_out = split_layer(self.outputs)

        # self.list_out = tf.split(
        #     self.outputs,
        #     [size_p, size_s],
        #     axis=1,
        #     name=output_name,
        # )
        self.primary_out = Dense(size_p, activation=None, name="out_1")(
            self.list_out[0]
        )
        self.secondary_out = Dense(size_s, activation=None, name="out_2")(
            self.list_out[0]
        )

        # manually create a placeholder for each small output layer
        # but it does not seem to work
        # self.list_out = [
        #     self.generate_output(self.size_inp[i], i + 1)(self.outputs)
        #     for i in range(len(self.size_inp))
        # ]

    def build_autoencoder(self, model_name="daenn"):
        """Build autoencoder model. Combine encoder and decoder together so that they are trained together.

        Args:
            model_name (str, optional): Name of model. Defaults to "daenn".
        """
        # Method 1
        # self.autoencoder = Model(inputs=self.inputs, outputs=self.outputs, name=model_name)

        # Method 2
        self.autoencoder = Model(
            inputs=[self.primary_inp] + self.secondary_inp,
            outputs=[self.primary_out, self.secondary_out],
            name=model_name,
        )

        # It seems that using multiple separate inputs and outputs for a model is wrong
        # because loss calculation will be applied to each individual output layer
        # but we need all input data in the same layer and takes once only loss computation
        #
        # self.autoencoder = Model(inputs=self.list_inp, outputs=self.list_out, name=model_name)

    def custom_loss_1(self, main_loss, penalty_loss, gamma=DEFAULT_GAMMA):
        """Method 1: encapsulation

        Use the closure to make a custom loss be able to receive additional arguments.
        But keep in mind that this could yield a potential problem when loading a model.

        Another workaround is to use 'model.add_loss' or write a new custom loss class using
        'tf.keras.losses.Loss' as a parent and wrap call(self, y_true, y_pred) function which does
        math operation with custom losses and return the value of loss function for optimization.
        See this answer https://stackoverflow.com/a/66486573/6596684 for more details.

        Args:
            main_loss (func): Main loss
            penalty_loss (func): Loss-like penalty function
            gamma (float): Weight for scaling down the impact of a loss function. Defaults to DEFAULT_GAMMA.

        Returns:
            _loss (tensor): Return values from the closure function
        """
        # For backward compatable
        # tf.compat.v1.enable_eager_execution()

        def _loss(y_true, y_pred):
            split_index = functools.reduce(
                lambda x, y: x + y, self.size_inp[: self.num_primary]
            )
            # Debug
            # y_true_penalty = y_true[split_index:]
            # y_pred_penalty = y_pred[split_index:]

            return (gamma * main_loss(y_true, y_pred)) - (
                (1 - gamma) * penalty_loss(y_true, y_pred)
            )

        return _loss

    def custom_loss_2(
        self, y_true, y_pred, main_loss, penalty_loss, gamma=DEFAULT_GAMMA
    ):
        """Method 2: add_loss

        Custom loss for model.add_loss(). add_loss creates loss as tensor, not function,
        which can take other variables as argument.

        Args:
            y_true (array): Array of reference values (targets)
            y_pred (array): Array of prediction values
            main_loss (func): Main loss
            penalty_loss (func): Loss-like penalty function
            gamma (float): Weight for scaling down the impact of a loss function. Defaults to DEFAULT_GAMMA.

        Returns:
            tensor: Return values from the closure function
        """
        return (
            (gamma * main_loss(y_true, y_pred))
            - ((1 - gamma) * penalty_loss(y_true, y_pred))
            # + tf.keras.backend.reduce_mean(self.latent)
        )

    def custom_loss_3(self, gamma=DEFAULT_GAMMA):
        """Method 3: external loss

        Define a class of loss and call it

        Args:
            gamma (float): Weight for scaling down the impact of a loss function. Defaults to DEFAULT_GAMMA.

        Returns:
            loss.CustomLoss (class): custom loss object
        """
        return loss.CustomLoss(self.main_loss, self.penalty_loss, self.latent, gamma)

    def loss_1(self, main_loss):
        """Custom main loss for primary dataset. Loss = (Loss * its loss_weight).

        Args:
            main_loss (func): Main loss

        Returns:
            _loss_1 (func): Return values from the closure function
        """

        def _loss_1(y_true, y_pred):
            return main_loss(y_true, y_pred)

        return _loss_1

    def loss_2(self, penalty_loss):
        """Custom penalty loss for secondary dataset. Loss = (Loss * its loss_weight).

        Maximization is used for this loss.

        Args:
            penalty_loss (func): Loss-like penalty function

        Returns:
            _loss_2 (func): Return values from the closure function
        """

        def _loss_2(y_true, y_pred):
            return 1 / penalty_loss(y_true, y_pred)

        return _loss_2

    def compile_model(self, optimizer, main_loss, penalty_loss, loss_weights):
        """Compile neural network model.

        Args:
            optimizer (str): Name of optimizer
            main_loss (func): Main loss
            penalty_loss (func): Loss-like penalty function
            loss_weights (list): Weight for each loss
        """
        self.optimizer = optimizer
        self.main_loss = main_loss
        self.penalty_loss = penalty_loss
        self.loss_weights = loss_weights

        # ------------------------------------------
        # Calling loss customization function/object
        # ------------------------------------------
        # self.autoencoder.add_loss(
        #     self.custom_loss_2(self.inputs, self.latent, self.main_loss, self.penalty_loss, gamma=DEFAULT_GAMMA)
        # ) # uncomment this line to use method 2

        self.autoencoder.compile(
            optimizer=self.optimizer,
            loss={
                "out_1": self.loss_1(self.main_loss),
                "out_2": self.loss_2(self.penalty_loss),
            },
            # loss=self.custom_loss_1(
            #     self.main_loss, self.penalty_loss, gamma=DEFAULT_GAMMA
            # ),  # method 1
            # loss=None,  # method 2
            # loss=self.custom_loss_3(gamma=DEFAULT_GAMMA),  # method 3
            loss_weights=self.loss_weights,
            metrics=["mse", "mse"],
            # run_eagerly=True,
        )

    def train_model(
        self,
        num_epoch,
        batch_size,
        save_every_n_epoch,
        verbose=1,
        log_dir="./logs/",
        out_dir="./",
    ):
        """Train model. Save model (latent space) every N-th epoch defined by the user

        Args:
            num_epoch (int): Number of epochs
            batch_size (int): Batch size
            save_every_n_epoch (int): Number of trainings (Defaults to 10 if not defined by the user)
            verbose (int): Level of prining information. Defaults to 1.
            log_dir (str): Directory to save model logs. Defaults to "./logs/".
            out_dir (str): Directory to save visualization outputs. Defaults to "./".
        """
        # stack all dataset into a single array since both input layer and output layer are a single layer
        # self.train_set = tf.concat(self.train_set, axis=1)
        # self.test_set = tf.concat(self.test_set, axis=1)

        self.train_set = [
            tf.concat(self.train_set[: self.num_primary], axis=1),
            tf.concat(self.train_set[self.num_primary :], axis=1),
        ]
        self.test_set = [
            tf.concat(self.test_set[: self.num_primary], axis=1),
            tf.concat(self.test_set[self.num_primary :], axis=1),
        ]

        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.save_every_n_epoch = save_every_n_epoch
        if self.batch_size == 0:
            self.batch_size = None

        # TF Board
        # You can use tensorboard to visualize TensorFlow runs and graphs.
        # e.g. 'tensorflow --logdir ./log'
        tbCallBack = TensorBoard(
            log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True
        )

        # Save model during the training
        class SaveLatentSpaceEveryNepoch(Callback):
            """A callback to save latent space as a figure at every N epoch"""

            def __init__(self, every_n_epoch, x_train, y_train, output_name):
                self.every_n_epoch = every_n_epoch
                self.x_train = x_train
                self.y_train = y_train
                self.output_name = output_name

            def on_epoch_end(self, epoch, logs=None):
                if (epoch % self.every_n_epoch) == 0:
                    logging.info(f"Save model at epoch no. {epoch}")
                    ev = ae_visual.encode_fig(
                        epoch,
                        self.model.get_layer("concatenate_1").input,
                        self.model.get_layer("dense_2").output,
                        self.x_train,
                        self.y_train,
                        folder=self.output_name,
                    )

        saveCallback = [
            SaveLatentSpaceEveryNepoch(
                self.save_every_n_epoch, self.train_set, self.train_set, out_dir
            )
        ]

        # Training begins here
        self.history = self.autoencoder.fit(
            x=self.train_set,  # input
            y=self.train_set,  # target
            shuffle=True,
            # validation_split=0.20,
            validation_data=(self.test_set, self.test_set),
            epochs=self.num_epoch,
            batch_size=self.batch_size,
            verbose=verbose,
            callbacks=[tbCallBack, saveCallback],
        )

    def autoencoder_predict(self, test_set):
        """Generate predictions using a trained autoencoder

        Args:
            test_set (array): Sample

        Returns:
            array: Prediction output
        """
        return self.autoencoder.predict(test_set)

    @staticmethod
    def save_summary(model, save_dir):
        """Save summary of a model

        Args:
            model (Class): Model to save
            save_dir (str): Output directory
        """
        model.summary()
        # save summary to a text file
        path = Path(save_dir) / "model_summary.txt"
        with open(path, "w") as f:
            with redirect_stdout(f):
                model.summary()

    @staticmethod
    def save_model(model, path_output):
        """Save model either in keras format or in SavedModel format (low-level)

        Args:
            model (Class): Model to save
            path_output (str): Path of output to save
        """
        try:
            tf.keras.models.save_model(
                model, path_output + ".keras", overwrite=True, include_optimizer=True
            )
            logging.info(f"Model saved in Keras format: {path_output}.keras")
        except:
            logging.error(f"Failed to save model in Keras format: {path_output}.keras")
            try:
                model.export(path_output)
                logging.info(f"Model exported to: {path_output}")
            except:
                logging.error(
                    f"Failed to export model in SavedModel format: {path_output}"
                )

    @staticmethod
    def save_graph(model, file_name="graph", save_dir=None, dpi=DEFAULT_DPI):
        """Plot model and save it as image

        Args:
            model (Class): Model to save
            file_name (str, optional): Name of model graph. Defaults to "graph".
            save_dir (str, optional): Output directory. Defaults to current working directory.
            dpi (int, optional): Image resolution. Defaults to DEFAULT_DPI.
        """
        if save_dir is None:
            save_dir = os.getcwd()

        output_path = Path(save_dir) / f"{file_name}.png"

        plot_model(
            model,
            to_file=str(output_path),
            show_shapes=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=False,
            dpi=dpi,
        )


def save_loss_history(model, config: DAENNConfig, out_autoencoder: str) -> None:
    """Save loss history plots for training visualization.

    Args:
        model: Trained autoencoder model with history
        config: Configuration object containing output settings
        out_autoencoder: Path to autoencoder output directory
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))

    # Total loss
    ax1.set_title("Total loss")
    ax1.plot(model.history.history["loss"], label="train")
    ax1.plot(model.history.history["val_loss"], label="test")
    ax1.set_ylabel("loss")
    ax1.set_xlabel("epoch")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Main loss (primary dataset)
    ax2.set_title("Main loss (primary dataset)")
    ax2.plot(model.history.history["out_1_loss"], label="train")
    ax2.plot(model.history.history["val_out_1_loss"], label="test")
    ax2.set_ylabel("loss")
    ax2.set_xlabel("epoch")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # Penalty loss (secondary dataset)
    ax3.set_title("Penalty loss (secondary dataset)")
    ax3.plot(model.history.history["out_2_loss"], label="train")
    ax3.plot(model.history.history["val_out_2_loss"], label="test")
    ax3.set_ylabel("loss")
    ax3.set_xlabel("epoch")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)

    # Adjust layout and save
    plt.tight_layout()
    save_path = Path(out_autoencoder) / config.output.loss_plot
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    logging.info(f"Loss history plot saved to {save_path}")

    if config.settings.show_loss:
        plt.show()


def save_metrics_history(model, config: DAENNConfig, out_autoencoder: str) -> None:
    """Save metrics history plots for training visualization.

    Args:
        model: Trained autoencoder model with history
        config: Configuration object containing output settings
        out_autoencoder: Path to autoencoder output directory
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Accuracy for main loss (primary dataset)
    ax1.plot(model.history.history["out_1_mse"], label="train")
    ax1.plot(model.history.history["val_out_1_mse"], label="test")
    ax1.set_title("Accuracy for main loss (primary dataset)")
    ax1.set_ylabel("MSE")
    ax1.set_xlabel("epoch")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Accuracy for penalty loss (secondary dataset)
    ax2.plot(model.history.history["out_2_mse"], label="train")
    ax2.plot(model.history.history["val_out_2_mse"], label="test")
    ax2.set_title("Accuracy for penalty loss (secondary dataset)")
    ax2.set_ylabel("MSE")
    ax2.set_xlabel("epoch")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # Adjust layout and save
    plt.tight_layout()
    save_path = Path(out_autoencoder) / config.output.metrics_plot
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    logging.info(f"Metrics history plot saved to {save_path}")

    if config.settings.show_metrics:
        plt.show()


def load_and_validate_config(input_file: str) -> DAENNConfig:
    """Load and validate configuration from JSON file.

    Args:
        input_file (str): Path to the JSON configuration file

    Returns:
        DAENNConfig: Validated configuration object

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If configuration is invalid
    """
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f'No such file "{input_file}"')

    try:
        json_data = util.load_json(input_file)
        config = DAENNConfig.from_json(json_data)
        logging.info("Configuration loaded and validated successfully")
        return config
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        raise


def main():
    info = "Autoencoder for learning collective variables from features."
    parser = argparse.ArgumentParser(description=info)
    parser.add_argument(
        "-i",
        "--input",
        metavar="INPUT",
        type=str,
        required=True,
        help="Input file (JSON) defining configuration, setting, parameters.",
    )

    args = parser.parse_args()

    # Load and validate configuration
    try:
        config = load_and_validate_config(args.input)
    except (FileNotFoundError, ValueError) as e:
        logging.error(str(e))
        sys.exit(1)

    # ========================================

    # sys.stdout = open("stdout_daenn_{:%Y_%m_%d_%H_%M_%S}.txt".format(datetime.now()), "w")

    logging.info("=" * 30 + " Program started " + "=" * 30)
    logging.info(f"Project: {config.project.name}")
    logging.info(
        "Date {:%d/%m/%Y}".format(datetime.now())
        + " at {:%H:%M:%S}".format(datetime.now())
    )

    if config.project.neural_network.lower() != "daenn":
        logging.error(f"This is a DAENN trainer, not {config.project.neural_network}.")
        sys.exit(1)

    ################
    # Check datasets
    ################
    # concatenate two lists of primary dataset (for main loss) and
    # secondary dataset (for penalty loss) into a single list for convenient preprocessing
    all_dataset = config.dataset.primary + config.dataset.secondary

    # Key
    try:
        dataset_keys = config.dataset.keys
        no_keys = dataset_keys is None
        if not no_keys:
            assert len(all_dataset) == len(
                dataset_keys
            ), "Total number of datasets and number of keys provided in the input file are not the same"
    except:
        no_keys = True

    # Extract features (input)
    if no_keys:
        logging.warning(
            "No npz keys specified, the first key found in array.files is automatically used."
        )
        dataset_arr_raw = [np.load(i) for i in all_dataset]
        dataset_arr = [i[i.files[0]] for i in dataset_arr_raw]
    else:
        dataset_arr = [np.load(i)[j] for i, j in zip(all_dataset, dataset_keys)]

    ###############
    # Preprocessing
    ###############
    # Use FP32 for speeding training and reducing precision error
    dataset_arr = [i.astype(FLOAT_PRECISION) for i in dataset_arr]

    logging.info("=== Shape of dataset before splitting ===")
    for i, j in enumerate(dataset_arr):
        logging.info(f"{i+1}. Dataset: {j.shape}")

    # Split dataset
    if not config.dataset.split:
        logging.error("Supports only spitting. Please set split to true.")
        sys.exit(1)
    train_arr, test_arr = [], []
    for i, j in enumerate(dataset_arr):
        train, test = train_test_split(
            j,
            train_size=config.dataset.split_ratio,
            shuffle=config.dataset.shuffle,
            random_state=RANDOM_STATE,
        )
        train_arr.append(train)
        test_arr.append(test)

    logging.info("=== Shape of dataset after splitting ===")
    for i, j in enumerate(train_arr):
        logging.info(f"{i+1}. Train: {j.shape} & Test: {test_arr[i].shape}")

    # Normalization
    if float(config.dataset.max_scale) == 0.0:
        try:
            max_scale_train = [i.max() for i in train_arr]
            max_scale_test = [i.max() for i in test_arr]
            logging.warning(
                f"As max_scale is set to {config.dataset.max_scale}, use maximum value in a dataset for scaling"
            )
        except:
            logging.error("Cannot determine maximum scale")
            sys.exit(1)
    else:
        max_scale_train = [config.dataset.max_scale for i in train_arr]
        max_scale_test = [config.dataset.max_scale for i in test_arr]

    try:
        # train_set = (train_set.astype(FLOAT_PRECISION) - normalize_scale) / max_scale
        train_arr = [
            (j - config.dataset.normalize_scale) / max_scale_train[i]
            for i, j in enumerate(train_arr)
        ]
        test_arr = [
            (j - config.dataset.normalize_scale) / max_scale_test[i]
            for i, j in enumerate(test_arr)
        ]
    except:
        logging.error("Normalization failed. Please check scaling parameters")
        sys.exit(1)

    ########################
    # Check layer parameters
    ########################
    error_assert = (
        "Number of units/hidden layer [units] is not equal to number of activation functions "
        + "[act_funcs]. Please check your DAENN input file!"
    )
    assert len(config.network.units) == len(config.network.act_funcs), error_assert

    # Activation function
    tf_act_func = getmembers(tf.keras.activations, isfunction)
    avail_act_func = [i[0] for i in tf_act_func]

    def check_act_func(f):
        if f in avail_act_func:
            return f
        if f.lower() == "leakyrelu":
            logging.warning("LeakyReLU is used as a loss function.")
            return LeakyReLU(alpha=0.2)
        else:
            err = (
                f"Activation function that you specified, {f}, is not supported."
                + f"Available functions are {avail_act_func}"
            )
            logging.error(err)
            sys.exit(1)

    user_act_func = list(map(check_act_func, config.network.act_funcs))

    ######
    # Loss
    ######
    tf_loss = dict(getmembers(tf.keras.losses, isfunction))

    # Main loss
    if config.model.main_loss in tf_loss.keys():
        main_loss = tf_loss[config.model.main_loss]
    elif config.model.main_loss in list(vars(loss).keys()):
        main_loss = vars(loss)[config.model.main_loss]
    else:
        err = (
            f"Loss function you specified for main_loss, {config.model.main_loss}, is not supported."
            + f"Available losses are {tf_loss.keys()} and DeepCV losses are defined in loss.py"
        )
        logging.error(err)
        sys.exit(1)

    # Penalty loss (loss-like penalty function for DAENN)
    if config.model.penalty_loss in tf_loss.keys():
        penalty_loss = tf_loss[config.model.penalty_loss]
    elif config.model.penalty_loss in list(vars(loss).keys()):
        penalty_loss = vars(loss)[config.model.penalty_loss]
    else:
        err = (
            f"Loss function you specified for penalty_loss, {config.model.penalty_loss}, is not supported."
            + f"Available TF losses are {tf_loss.keys()} and DeepCV losses are defined in loss.py"
        )
        logging.error(err)
        sys.exit(1)

    ##############
    # Check output
    ##############
    # Make sure that the top output directory exists
    out_parent = os.path.abspath(config.output.out_dir)
    try:
        Path(out_parent).mkdir(parents=True, exist_ok=True)
    except:
        logging.error(
            f"Can't create output directory you specified, {config.output.out_dir}!"
        )
        sys.exit(1)
    # Create output directory for autoencoder, encoder and decoder
    out_autoencoder = out_parent + "/autoencoder/"
    Path(out_autoencoder).mkdir(parents=True, exist_ok=True)
    out_encoder = out_parent + "/encoder/"
    Path(out_encoder).mkdir(parents=True, exist_ok=True)
    out_decoder = out_parent + "/decoder/"
    Path(out_decoder).mkdir(parents=True, exist_ok=True)

    ##############################################################
    # Build, compile and train encoder & decoder models separately
    ##############################################################
    model = Autoencoder()
    model.add_dataset(
        train_arr, test_arr, len(config.dataset.primary), len(config.dataset.secondary)
    )

    # Detect GPUs
    gpus = tf.config.list_physical_devices("GPU")

    if config.performance.enable_gpu and len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()  # Multi-GPU
        logging.info("Using Multi-GPU (MirroredStrategy)")
    else:
        strategy = tf.distribute.get_strategy()  # Single GPU or CPU
        logging.info("Using Single GPU/CPU")

    with strategy.scope():
        model.build_network(units=config.network.units, act_funcs=user_act_func)
        model.build_autoencoder()
        model.compile_model(
            config.model.optimizer, main_loss, penalty_loss, config.model.loss_weights
        )

    # show model info
    if config.settings.show_summary:
        model.save_summary(model.autoencoder, out_autoencoder)

    # Train model
    out_tb = Path(out_parent) / config.output.out_tb
    model.train_model(
        config.model.num_epoch,
        config.model.batch_size,
        config.model.save_every_n_epoch,
        config.settings.verbosity,
        out_tb,
        out_autoencoder,
    )
    logging.info("Congrats! Training model is completed.")

    # Test model
    test_set = [
        tf.concat(test_arr[: len(config.dataset.primary)], axis=1),
        tf.concat(test_arr[len(config.dataset.primary) :], axis=1),
    ]
    assert model.autoencoder_predict(
        test_set
    ), "Failed to make a prediction. Check your network architecture again!"

    ########################
    # Save model and outputs
    ########################
    if config.settings.save_model:
        model.save_model(
            model.autoencoder, Path(out_autoencoder) / config.output.out_model
        )
        logging.info(f"Model saved to {out_parent}")

    if config.settings.save_weights:
        path = Path(out_autoencoder) / config.output.out_weights
        model.autoencoder.save_weights(path, overwrite=True)
        logging.info(f"Weights of model saved to {path}")

    if config.settings.save_weights_npz:
        filename = os.path.splitext(config.output.out_weights_npz)[0]
        path = Path(out_autoencoder) / (filename + ".npz")
        savez = dict()
        for i, layer in enumerate(model.autoencoder.layers):
            if layer.get_weights():
                savez["layer" + str(i + 1)] = layer.get_weights()[0]
        np.savez_compressed(path, **savez)
        logging.info(f"Weights of model saved to {path}")

    if config.settings.save_biases_npz:
        filename = os.path.splitext(config.output.out_biases_npz)[0]
        path = Path(out_autoencoder) / (filename + ".npz")
        savez = dict()
        for i, layer in enumerate(model.autoencoder.layers):
            if layer.get_weights():
                savez["layer" + str(i + 1)] = layer.get_weights()[1]
        np.savez_compressed(path, **savez)
        logging.info(f"Biases of model saved to {path}")

    if config.settings.save_graph:
        model.save_graph(model.autoencoder, model.autoencoder.name, out_autoencoder)
        logging.info(
            f"Directed graph of model saved to subfolder in {os.path.abspath(config.output.out_dir)}"
        )

    # Save training history visualizations
    if config.settings.save_loss:
        save_loss_history(model, config, out_autoencoder)
    if config.settings.save_metrics:
        save_metrics_history(model, config, out_autoencoder)

    logging.info("=" * 30 + " DONE " + "=" * 30)

    # sys.stdout.close()


if __name__ == "__main__":
    main()
