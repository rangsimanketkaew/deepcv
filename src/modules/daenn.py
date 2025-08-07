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


class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()

    def add_dataset(self, train_set, test_set, num_primary, num_secondary):
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

    def build_encoder(self, name="encoder"):
        """Build encoder model. An encoder is a neural network (it can be any type of network, e.g., FC, CNN,
        RNN, etc) that takes the input, and output a feature map/vector/tensor. These feature vector hold the
        information, the features, that represents the input.

        Args:
            name (str, optional): Name of model. Defaults to "encoder".
        """
        # There are two ways to create an encoder with different input branchs.
        #  1) Use a list of sepatate input layers
        # self.encoder = Model(inputs=self.list_inp, outputs=self.latent, name=name)
        #  2) Use already-merged input so that the encoder can be later imported by TF graph loader
        #     and it will take only one dataset which combines already input size already
        self.encoder = Model(
            inputs=[self.primary_inp] + self.secondary_inp,
            outputs=self.latent,
            name=name,
        )

    def build_decoder(self, model_name="decoder", output_name="decoder_output"):
        """Build decoder model. The decoder is a network (usually the same network structure as encoder but in
        opposite orientation) that takes the feature vector from the encoder, and gives the best closest match
        to the actual input or intended output.

        Args:
            model_name (str, optional): Name of model. Defaults to "decoder".
            output_name (str, optional): Name of the output layer. Defaults to "decoder_output".
        """
        # We can't use a latent (Tensor) dense layer as an input layer of the decoder because
        # the input of Model class accepts only Input layer. We therefore need to re-construct
        # the decoder based on the second half of Autoencoder.
        decoder_input = Input(shape=(self.latent.shape[1],), name="decoder_input")
        decoded1 = Dense(self.units_4, activation=self.func_4, use_bias=True)(
            decoder_input
        )
        decoded2 = Dense(self.units_5, activation=self.func_5, use_bias=True)(decoded1)
        outputs = Dense(self.inputs.shape[1], activation=None, use_bias=True)(
            decoded2
        )  # reconstruct input

        split_layer = Lambda(
            lambda x: tf.split(x, self.size_inp, axis=1, name=output_name)
        )
        output_split = split_layer(outputs)
        self.decoder = Model(
            inputs=decoder_input,
            outputs=[output_split],
            name=model_name,
        )

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

    def custom_loss_1(self, main_loss, penalty_loss, gamma=0.8):
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
            gamma (float): Weight for scaling down the impact of a loss function. Defaults to 0.8.

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

    def custom_loss_2(self, y_true, y_pred, main_loss, penalty_loss, gamma=0.8):
        """Method 2: add_loss

        Custom loss for model.add_loss(). add_loss creates loss as tensor, not function,
        which can take other variables as argument.

        Args:
            y_true (array): Array of reference values (targets)
            y_pred (array): Array of prediction values
            main_loss (func): Main loss
            penalty_loss (func): Loss-like penalty function
            gamma (float): Weight for scaling down the impact of a loss function. Defaults to 0.8.

        Returns:
            tensor: Return values from the closure function
        """
        return (
            (gamma * main_loss(y_true, y_pred))
            - ((1 - gamma) * penalty_loss(y_true, y_pred))
            # + tf.keras.backend.reduce_mean(self.latent)
        )

    def custom_loss_3(self, gamma):
        """Method 3: external loss

        Define a class of loss and call it

        Args:
            gamma (float): Weight for scaling down the impact of a loss function. Defaults to 0.8.

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
        #     self.custom_loss_2(self.inputs, self.latent, self.main_loss, self.penalty_loss, alpha=0.8)
        # ) # uncomment this line to use method 2

        self.autoencoder.compile(
            optimizer=self.optimizer,
            loss={
                "out_1": self.loss_1(self.main_loss),
                "out_2": self.loss_2(self.penalty_loss),
            },
            # loss=self.custom_loss_1(self.main_loss, self.penalty_loss, gamma=0.8),  # method 1
            # loss=None, # method 2
            # loss=self.custom_loss_3(alpha=0.8), # method 3
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

    def encoder_predict(self, input_sample):
        """Generate predictions using encoder

        Args:
            input_sample (array): Input sample

        Returns:
            array: Prediction output
        """
        return self.encoder.predict(input_sample)

    def decoder_predict(self, encoded_sample):
        """Generate predictions using decoder

        Args:
            encoded_sample (array): Encoded sample from the latent layer of encoder

        Returns:
            array: Prediction output
        """
        return self.decoder.predict(encoded_sample)

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
        path = save_dir + "/" + "model_summary.txt"
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
                logging.error(f"Failed to export model in SavedModel format: {path_output}")

    @staticmethod
    def save_graph(model, file_name="graph", save_dir=None, dpi=192):
        """Plot model and save it as image

        Args:
            model (Class): Model to save
            file_name (str, optional): Name of model graph. Defaults to "graph".
            save_dir (str, optional): Output directory. Defaults to current working directory.
            dpi (int, optional): Image resolution. Defaults to 192.
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

    if not os.path.isfile(args.input):
        logging.error('No such file "' + str(args.input) + '"')
        sys.exit(1)

    # Load data from JSON input file
    json = util.load_json(args.input)
    project = json["project"]["name"]
    neural_network = json["project"]["neural_network"]
    # ---------
    primary_data = json["dataset"]["primary"]
    secondary_data = json["dataset"]["secondary"]
    split = json["dataset"]["split"]
    split_ratio = json["dataset"]["split_ratio"]
    shuffle = json["dataset"]["shuffle"]
    normalize_scale = json["dataset"]["normalize_scale"]
    max_scale = json["dataset"]["max_scale"]
    # ---------
    optimizer = json["model"]["optimizer"]
    main_loss = json["model"]["main_loss"]
    penalty_loss = json["model"]["penalty_loss"]
    loss_weights = json["model"]["loss_weights"]
    num_epoch = json["model"]["num_epoch"]
    batch_size = json["model"]["batch_size"]
    try:
        save_every_n_epoch = json["model"]["save_every_n_epoch"]
        save_every_n_epoch = int(save_every_n_epoch)
    except KeyError:
        save_every_n_epoch = int(num_epoch / 10)
        if save_every_n_epoch == 0:
            save_every_n_epoch = 10
    # ---------
    units = json["network"]["units"]
    act_funcs = json["network"]["act_funcs"]
    # ---------
    enable_gpu = json["performance"]["enable_gpu"]
    gpus = json["performance"]["gpus"]
    # ---------
    verbosity = json["settings"]["verbosity"]
    show_summary = json["settings"]["show_summary"]
    save_model = json["settings"]["save_model"]
    save_weights = json["settings"]["save_weights"]
    save_weights_npz = json["settings"]["save_weights_npz"]
    save_biases_npz = json["settings"]["save_biases_npz"]
    save_graph = json["settings"]["save_graph"]
    save_loss = json["settings"]["save_loss"]
    show_loss = json["settings"]["show_loss"]
    save_metrics = json["settings"]["save_metrics"]
    show_metrics = json["settings"]["show_metrics"]
    # ---------
    out_dir = json["output"]["out_dir"]
    out_tb = json["output"]["out_tb"]
    out_model = json["output"]["out_model"]
    out_weights = json["output"]["out_weights"]
    out_weights_npz = json["output"]["out_weights_npz"]
    out_biases_npz = json["output"]["out_biases_npz"]
    loss_plot = json["output"]["loss_plot"]
    metrics_plot = json["output"]["metrics_plot"]

    # ========================================

    # sys.stdout = open("stdout_daenn_{:%Y_%m_%d_%H_%M_%S}.txt".format(datetime.now()), "w")

    logging.info("=" * 30 + " Program started " + "=" * 30)
    logging.info(f"Project: {project}")
    logging.info(
        "Date {:%d/%m/%Y}".format(datetime.now())
        + " at {:%H:%M:%S}".format(datetime.now())
    )

    if neural_network.lower() != "daenn":
        logging.error(f"This is a DAENN trainer, not {neural_network}.")
        sys.exit(1)

    ################
    # Check datasets
    ################
    # concatenate two lists of primary dataset (for main loss) and
    # secondary dataset (for penalty loss) into a single list for convenient preprocessing
    all_dataset = primary_data + secondary_data

    # Key
    try:
        dataset_keys = json["dataset"]["keys"]
        no_keys = False
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
    dataset_arr = [i.astype(np.float32) for i in dataset_arr]

    logging.info("=== Shape of dataset before splitting ===")
    for i, j in enumerate(dataset_arr):
        logging.info(f"{i+1}. Dataset: {j.shape}")

    # Split dataset
    if not split:
        logging.error("Supports only spitting. Please set split to true.")
        sys.exit(1)
    train_arr, test_arr = [], []
    for i, j in enumerate(dataset_arr):
        train, test = train_test_split(
            j, train_size=split_ratio, shuffle=shuffle, random_state=42
        )
        train_arr.append(train)
        test_arr.append(test)

    logging.info("=== Shape of dataset after splitting ===")
    for i, j in enumerate(train_arr):
        logging.info(f"{i+1}. Train: {j.shape} & Test: {test_arr[i].shape}")

    # Normalization
    if float(max_scale) == 0.0:
        try:
            max_scale_train = [i.max() for i in train_arr]
            max_scale_test = [i.max() for i in test_arr]
            logging.warning(
                f"As max_scale is set to {max_scale}, use maximum value in a dataset for scaling"
            )
        except:
            logging.error("Cannot determine maximum scale")
            sys.exit(1)
    else:
        max_scale_train = [max_scale for i in train_arr]
        max_scale_test = [max_scale for i in test_arr]

    try:
        # train_set = (train_set.astype(np.float32) - normalize_scale) / max_scale
        train_arr = [
            (j - normalize_scale) / max_scale_train[i] for i, j in enumerate(train_arr)
        ]
        test_arr = [
            (j - normalize_scale) / max_scale_test[i] for i, j in enumerate(test_arr)
        ]
    except:
        logging.error("Normalization failed. Please check scaling parameters")
        sys.exit(1)

    # Train on GPU?
    if not enable_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    ########################
    # Check layer parameters
    ########################
    error_assert = (
        "Number of units/hidden layer [units] is not equal to number of activation functions "
        + "[act_funcs]. Please check your DAENN input file!"
    )
    assert len(units) == len(act_funcs), error_assert

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

    user_act_func = list(map(check_act_func, act_funcs))

    ######
    # Loss
    ######
    tf_loss = dict(getmembers(tf.keras.losses, isfunction))

    # Main loss
    if main_loss in tf_loss.keys():
        main_loss = tf_loss[main_loss]
    elif main_loss in list(vars(loss).keys()):
        main_loss = vars(loss)[main_loss]
    else:
        err = (
            f"Loss function you specified for main_loss, {main_loss}, is not supported."
            + f"Available losses are {tf_loss.keys()} and DeepCV losses are defined in loss.py"
        )
        logging.error(err)
        sys.exit(1)

    # Penalty loss (loss-like penalty function for DAENN)
    if penalty_loss in tf_loss.keys():
        penalty_loss = tf_loss[penalty_loss]
    elif penalty_loss in list(vars(loss).keys()):
        penalty_loss = vars(loss)[penalty_loss]
    else:
        err = (
            f"Loss function you specified for penalty_loss, {penalty_loss}, is not supported."
            + f"Available TF losses are {tf_loss.keys()} and DeepCV losses are defined in loss.py"
        )
        logging.error(err)
        sys.exit(1)

    ##############
    # Check output
    ##############
    # Make sure that the top output directory exists
    out_parent = os.path.abspath(out_dir)
    try:
        Path(out_parent).mkdir(parents=True, exist_ok=True)
    except:
        logging.error(f"Can't create output directory you specified, {out_dir}!")
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
    model.add_dataset(train_arr, test_arr, len(primary_data), len(secondary_data))
    # Check if multi-GPU parallelization is possible
    if gpus == 1:
        model.build_network(units=units, act_funcs=user_act_func)
        model.build_autoencoder()
        model.compile_model(optimizer, main_loss, penalty_loss, loss_weights)
    elif gpus > 1:
        try:
            logging.warning("Training on multi-GPU on a single machine")
            # Use all available GPUs
            strategy = tf.distribute.MirroredStrategy(devices=None)
            with strategy.scope():
                model.build_network(units=units, act_funcs=user_act_func)
                model.build_autoencoder()
                model.compile_model(optimizer, main_loss, penalty_loss, loss_weights)
        except:
            logging.warning("Cannot enable multi-GPUs support for Keras")
            model.build_network(units=units, act_funcs=user_act_func)
            model.build_autoencoder()
            model.compile_model(optimizer, main_loss, penalty_loss, loss_weights)
    else:
        err = "Number of GPUs must be equal to or greater than 1"
        logging.error(err)
        sys.exit(1)

    # Construct encoder and decoder separately as well
    model.build_encoder()
    model.build_decoder()

    # show model info
    if show_summary:
        model.save_summary(model.autoencoder, out_autoencoder)
        model.save_summary(model.encoder, out_encoder)
        model.save_summary(model.decoder, out_decoder)

    # Test prediction
    # test_arr_ = tf.concat(test_arr, axis=1)
    # encoded_test = model.encoder_predict(test_arr_)
    # decoded_test = model.decoder_predict(encoded_test)

    # Train model
    out_tb = out_parent + "/" + out_tb
    model.train_model(
        num_epoch, batch_size, save_every_n_epoch, verbosity, out_tb, out_autoencoder
    )
    logging.info("Congrats! Training model is completed.")

    # Test prediction
    # test_arr_ = tf.concat(test_arr, axis=1)
    # assert model.autoencoder_predict(
    #     test_arr_
    # ), "Failed to make a prediction. Check your network architecture again!"

    ########################
    # Save model and outputs
    ########################
    if save_model:
        model.save_model(model.autoencoder, out_autoencoder + "/" + out_model)
        model.save_model(model.encoder, out_encoder + "/" + out_model)
        model.save_model(model.decoder, out_decoder + "/" + out_model)
        logging.info(f"Models have been saved to {out_parent}")

    if save_weights:
        path = out_autoencoder + "/" + out_weights
        model.autoencoder.save_weights(path, overwrite=True)
        logging.info(f"Weights of model have been saved to {path}")

    if save_weights_npz:
        filename = os.path.splitext(out_weights_npz)[0]
        path = out_autoencoder + "/" + filename + ".npz"
        savez = dict()
        for i, layer in enumerate(model.autoencoder.layers):
            if layer.get_weights():
                savez["layer" + str(i + 1)] = layer.get_weights()[0]
        np.savez_compressed(path, **savez)
        logging.info(f"Weights of model have been saved to {path}")

    if save_biases_npz:
        filename = os.path.splitext(out_biases_npz)[0]
        path = out_autoencoder + "/" + filename + ".npz"
        savez = dict()
        for i, layer in enumerate(model.autoencoder.layers):
            if layer.get_weights():
                savez["layer" + str(i + 1)] = layer.get_weights()[1]
        np.savez_compressed(path, **savez)
        logging.info(f"Biases of model have been saved to {path}")

    if save_graph:
        model.save_graph(model.autoencoder, model.autoencoder.name, out_autoencoder)
        model.save_graph(model.encoder, model.encoder.name, out_encoder)
        model.save_graph(model.decoder, model.decoder.name, out_decoder)
        logging.info(
            f"Directed graphs of all models have been saved to subfolder in {os.path.abspath(out_dir)}"
        )

    # summarize history for loss and accuracy
    if save_loss:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))
        # fig.suptitle("Loss")

        ax1.set_title("Total loss")
        ax1.plot(model.history.history["loss"])
        ax1.plot(model.history.history["val_loss"])
        ax1.set_ylabel("loss")
        ax1.set_xlabel("epoch")
        ax1.label_outer()
        ax1.legend(["train", "test"], loc="upper right")

        ax2.set_title("Main loss (primary dataset)")
        ax2.plot(model.history.history["out_1_loss"])
        ax2.plot(model.history.history["val_out_1_loss"])
        ax2.set_ylabel("loss")
        ax2.set_xlabel("epoch")
        ax2.label_outer()
        ax2.legend(["train", "test"], loc="upper right")

        ax3.set_title("Penalty loss (secondary dataset)")
        ax3.plot(model.history.history["out_2_loss"])
        ax3.plot(model.history.history["val_out_2_loss"])
        ax3.set_ylabel("loss")
        ax3.set_xlabel("epoch")
        ax3.label_outer()
        ax3.legend(["train", "test"], loc="upper right")

        save_path = out_autoencoder + "/" + loss_plot
        fig.savefig(save_path)
        logging.info(f"Loss history plot has been saved to {save_path}")

        if show_loss:
            fig.show()

    if save_metrics:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        ax1.plot(model.history.history["out_1_mse"])
        ax1.plot(model.history.history["val_out_1_mse"])
        ax1.set_title("Accuracy for main loss (primary dataset)")
        ax1.set_ylabel("accuracy metric")
        ax1.set_xlabel("epoch")
        ax1.label_outer()
        ax1.legend(["train", "test"], loc="upper right")

        ax2.plot(model.history.history["out_2_mse"])
        ax2.plot(model.history.history["val_out_2_mse"])
        ax2.set_title("Accuracy for penalty loss (secondary dataset)")
        ax2.set_ylabel("accuracy metric")
        ax2.set_xlabel("epoch")
        ax2.label_outer()
        ax2.legend(["train", "test"], loc="upper right")

        save_path = out_autoencoder + "/" + metrics_plot
        plt.savefig(save_path)
        logging.info(f"Metric accuracy history plot has been saved to {save_path}")
        if show_metrics:
            fig.show()

    logging.info("=" * 30 + " DONE " + "=" * 30)

    # sys.stdout.close()


if __name__ == "__main__":
    main()
