"""
Deep Learning for Collective Variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
28/11/2020 : Rangsiman Ketkaew
"""

"""
Deep autoencoder neural net (DAENN) for learning collective variables from molecular representations
"""

import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import argparse
import logging

logging = logging.getLogger("DeepCV")

from datetime import datetime
from inspect import getmembers, isfunction
from utils import util  # needs to be loaded before calling TF

import functools
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

try:
    assert tf.test.is_built_with_gpu_support()
    assert tf.config.list_physical_devices("GPU")
except AssertionError:
    pass
else:
    util.limit_gpu_growth()

from tensorflow.keras.layers import Input, Dense, Concatenate, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from modules import layer, loss
from tools import ae_visual


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
        self.units_1, self.units_2, self.units_3, self.units_4, self.units_5 = layer_params["units"]
        self.func_1, self.func_2, self.func_3, self.func_4, self.func_5 = layer_params["act_funcs"]

        # ---------
        # Create initial placeholder layer for each input dataset
        # ---------
        self.size_inp = [i.shape[1] for i in self.train_set]
        self.list_inp = [self.generate_input(self.size_inp[i], i + 1) for i in range(len(self.size_inp))]

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
            self.primary_inp = Concatenate(axis=-1)(self.list_inp[: self.num_primary])  # Merge branches

        # merge again with the rest of placeholder layer (secondary dataset(s))
        self.secondary_inp = self.list_inp[self.num_primary :]
        self.inputs = Concatenate(axis=-1)([self.primary_inp] + self.secondary_inp)

        # ---------
        # Encoder: Inp --> H1 --> H2 --> H3
        # latent = feature representation
        # ---------
        self.encoded1 = Dense(self.units_1, activation=self.func_1, use_bias=True)(self.inputs)
        self.encoded2 = Dense(self.units_2, activation=self.func_2, use_bias=True)(self.encoded1)
        self.latent = Dense(self.units_3, activation=self.func_3, use_bias=True)(self.encoded2)

        # ---------
        # Decoder: H3 --> H4 --> H5 --> Out
        # output = reconstruction
        # ---------
        self.decoded1 = Dense(self.units_4, activation=self.func_4, use_bias=True)(self.latent)
        self.decoded2 = Dense(self.units_5, activation=self.func_5, use_bias=True)(self.decoded1)
        self.outputs = Dense(self.inputs.shape[1], activation=None, use_bias=True)(self.decoded2)

        # ---------
        # split output into sub-tensors
        # Output: Out --> O1 + O2 + ...
        # ---------
        # size of primary layer [ : a]
        size_p = self.primary_inp.shape.as_list()[1]
        # size of secondary layer [a : ]
        size_s = functools.reduce(lambda x, y: x + y, self.size_inp[self.num_primary :])

        self.list_out = tf.split(self.outputs, [size_p, size_s], axis=1, name=output_name,)
        self.primary_out = Dense(size_p, activation=None, name="out_1")(self.list_out[0])
        self.secondary_out = Dense(size_s, activation=None, name="out_2")(self.list_out[0])

        # manually create a placeholder for each small output layer
        # but it does not seem to work
        # self.list_out = [
        #     self.generate_output(self.size_inp[i], i + 1)(self.outputs)
        #     for i in range(len(self.size_inp))
        # ]

    def build_encoder(self, name="encoder"):
        """Build encoder model.

        Args:
            name (str, optional): Name of model. Defaults to "encoder".
        """
        self.encoder = Model(inputs=self.list_inp, outputs=self.latent, name=name)

    def build_decoder(self, model_name="decoder", output_name="decoder_output"):
        """Build decoder model.

        Args:
            model_name (str, optional): Name of model. Defaults to "decoder".
            output_name (str, optional): Name of the output layer. Defaults to "decoder_output".
        """
        # We can't use a latent (Tensor) dense layer as an input layer of the decoder because
        # the input of Model class accepts only Input layer. We therefore need to re-construct
        # the decoder based on the second half of Autoencoder.
        decoder_input = Input(shape=(self.latent.shape[1],), name="decoder_input")
        decoded1 = Dense(self.units_4, activation=self.func_4, use_bias=True)(decoder_input)
        decoded2 = Dense(self.units_5, activation=self.func_5, use_bias=True)(decoded1)
        outputs = Dense(self.inputs.shape[1], activation=None, use_bias=True)(decoded2)  # reconstruct input
        self.decoder = Model(
            inputs=decoder_input,
            outputs=[tf.split(outputs, self.size_inp, axis=1, name=output_name)],
            name=model_name,
        )

    def build_autoencoder(self, model_name="daenn"):
        """Build autoencoder model.

        Args:
            model_name (str, optional): Name of model. Defaults to "daenn".
        """
        # self.autoencoder = Model(inputs=self.inputs, outputs=self.outputs, name=model_name)
        self.autoencoder = Model(
            inputs=[self.primary_inp] + self.secondary_inp, outputs=[self.primary_out, self.secondary_out]
        )

        # It seems that using multiple separate inputs and outputs for a model is wrong
        # because loss calculation will be applied to each individual output layer
        # but we need all input data in the same layer and takes once only loss computation
        #
        # self.autoencoder = Model(inputs=self.list_inp, outputs=self.list_out, name=model_name)

    def parallel_gpu(self, gpus=1):
        """
        Parallelization with multi-GPU.

        Args:
            gpus (int): Number of GPUs. Defaults to 1.
        """
        if gpus > 1:
            try:
                from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model

                self.autoencoder = multi_gpu_model(self.autoencoder, gpus=gpus)
                logging.warning(">>> Training on multi-GPU on a single machine")
            except:
                logging.warning(">>> Cannot enable multi-GPUs support for Keras")

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
            gamma (_type_): Weight for scaling down the impact of a loss function. Defaults to 0.8.

        Returns:
            tensor: Return values from the closure function
        """
        # tf.compat.v1.enable_eager_execution()

        def _loss(y_true, y_pred):
            split_index = functools.reduce(lambda x, y: x + y, self.size_inp[: self.num_primary])
            y_true_penalty = y_true[split_index:]
            y_pred_penalty = y_pred[split_index:]

            return (gamma * main_loss(y_true, y_pred)) - ((1 - gamma) * penalty_loss(y_true, y_pred))

        return _loss

    def custom_loss_2(self, y_true, y_pred, main_loss, penalty_loss, gamma=0.8):
        """Method 2: add_loss
        
        Custom loss for model.add_loss(). add_loss creates loss as tensor, not function, 
        which can take other variables as argument.

        Args:
            main_loss (func): Main loss
            penalty_loss (func): Loss-like penalty function
            gamma (_type_): Special weight. Defaults to 0.8.

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

        Returns:
            class: custom loss object
        """
        return loss.CustomLoss(self.main_loss, self.penalty_loss, self.latent, gamma)

    def loss_1(self, main_loss):
        """Custom main loss for primary dataset. Loss = (Loss * its loss_weight).

        Args:
            main_loss (func): Main loss

        Returns:
            tensor: Return values from the closure function
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
            tensor: Return values from the closure function
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
            loss={"out_1": self.loss_1(self.main_loss), "out_2": self.loss_2(self.penalty_loss),},
            # loss=self.custom_loss_1(self.main_loss, self.penalty_loss, gamma=0.8),  # method 1
            # loss=None, # method 2
            # loss=self.custom_loss_3(alpha=0.8), # method 3
            loss_weights=self.loss_weights,
            metrics=["mse"],
            # run_eagerly=True,
        )

    def train_model(self, num_epoch, batch_size, verbose=1, log_dir="./logs/"):
        """Train model
        
        Args:
            num_epoch (int): Number of epochs
            batch_size (int): Batch size
            verbose (int): Level of prining information. Defaults to 1.
            log_dir (str): Directory to save model logs. Defaults to "./logs/".
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
        if self.batch_size == 0:
            self.batch_size = None

        # TF Board
        # You can use tensorboard to visualize TensorFlow runs and graphs.
        # e.g. 'tensorflow --logdir ./log'
        tbCallBack = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)

        # TQDM progress bar
        tqdm_callback = tfa.callbacks.TQDMProgressBar()

        # train N times
        for i in range(10):
            self.history = self.autoencoder.fit(
                x=self.train_set,  # input
                y=self.train_set,  # target
                shuffle=True,
                # validation_split=0.20,
                validation_data=(self.test_set, self.test_set),
                epochs=self.num_epoch,
                batch_size=self.batch_size,
                verbose=verbose,
                use_multiprocessing=True,
                callbacks=[
                    tqdm_callback,
                    # tbCallBack
                ],
            )
            # save latent space
            ae_visual.encode_fig(i + 1, self.autoencoder.get_layer("concatenate_1").input, self.autoencoder.get_layer("dense_2").output, self.train_set, self.train_set)

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

    def save_graph(self, model, file_name="graph", save_dir=os.getcwd(), dpi=192):
        """Plot model and save it as image

        Args:
            model (Class): Model to save
            file_name (str, optional): Name of model graph. Defaults to "graph".
            save_dir (str, optional): Output directory. Defaults to os.getcwd().
            dpi (int, optional): Image resolution. Defaults to 192.
        """
        plot_model(
            model,
            to_file=save_dir + "/" + file_name + ".png",
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
    train_separately = json["model"]["train_separately"]
    # ---------
    units = json["network"]["units"]
    act_funcs = json["network"]["act_funcs"]
    # ---------
    enable_gpu = json["performance"]["enable_gpu"]
    gpus = json["performance"]["gpus"]
    # ---------
    verbosity = json["settings"]["verbosity"]
    show_summary = json["settings"]["show_summary"]
    save_tb = json["settings"]["save_tb"]
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
    out_model = json["output"]["out_model"]
    out_weights = json["output"]["out_weights"]
    out_weights_npz = json["output"]["out_weights_npz"]
    out_biases_npz = json["output"]["out_biases_npz"]
    loss_plot = json["output"]["loss_plot"]
    metrics_plot = json["output"]["metrics_plot"]

    # ========================================

    # sys.stdout = open("stdout_daenn_{:%Y_%m_%d_%H_%M_%S}.txt".format(datetime.now()), "w")

    print("=" * 30 + " Program started " + "=" * 30)
    print(f"Project: {project}")
    print("Date {:%d/%m/%Y}".format(datetime.now()) + " at {:%H:%M:%S}".format(datetime.now()))

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
        logging.warning("No npz keys specified, the first key found in array.files is automatically used.")
        dataset_arr_raw = [np.load(i) for i in all_dataset]
        dataset_arr = [i[i.files[0]] for i in dataset_arr_raw]
    else:
        dataset_arr = [np.load(i)[j] for i, j in zip(all_dataset, dataset_keys)]

    ###############
    # Preprocessing
    ###############
    # Use FP32 for speeding training and reducing precision error
    dataset_arr = [i.astype(np.float32) for i in dataset_arr]

    print("=== Shape of dataset before splitting ===")
    for i, j in enumerate(dataset_arr):
        print(f">>> {i+1}. Dataset: {j.shape}")

    # Split dataset
    train_arr, test_arr = [], []
    for i, j in enumerate(dataset_arr):
        train, test = train_test_split(j, train_size=split_ratio, shuffle=True, random_state=42)
        train_arr.append(train)
        test_arr.append(test)

    print("=== Shape of dataset after splitting ===")
    for i, j in enumerate(train_arr):
        print(f">>> {i+1}. Train: {j.shape} & Test: {test_arr[i].shape}")

    # Normalization
    train_arr = [i / i.max() for i in train_arr]
    test_arr = [i / i.max() for i in test_arr]

    # Train on GPU?
    if not enable_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    ########################
    # Check layer parameters
    ########################
    assert len(units) == len(
        act_funcs
    ), "Number of units/hidden layer [units] is not equal to number of activation functions [act_funcs]. \
Please check your DAENN input file!"

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
                f"Activation function youspecified, {f}, is not supported."
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
    if not os.path.isdir(out_dir):
        logging.error(
            f"Output directory you specified, {out_dir}, does not exist. Please create this directory!"
        )
        sys.exit(1)

    ##############################################################
    # Build, compile and train encoder & decoder models separately
    ##############################################################
    if train_separately:
        model.build_encoder()
        model.build_decoder()

        # Show model info
        if show_summary:
            model.encoder.summary()
            model.decoder.summary()

        # Test prediction
        test_arr_ = tf.concat(test_arr, axis=1)
        encoded_test = model.encoder_predict(test_arr_)
        decoded_test = model.decoder_predict(encoded_test)

        # Save TF graph
        if save_graph:
            model.save_graph(model.encoder, model.encoder.name, out_dir)
            model.save_graph(model.decoder, model.decoder.name, out_dir)

    ######################################
    # Build, compile and train DAENN model
    ######################################
    model = Autoencoder()
    model.add_dataset(train_arr, test_arr, len(primary_data), len(secondary_data))
    model.build_network(units=units, act_funcs=user_act_func)
    model.build_autoencoder()
    model.compile_model(optimizer, main_loss, penalty_loss, loss_weights)
    # show model info
    if show_summary:
        model.autoencoder.summary()
    # Train model
    model.train_model(num_epoch, batch_size, verbosity, save_tb)
    logging.info("Congrats! Training model is completed.")

    # Test prediction
    # test_arr_ = tf.concat(test_arr, axis=1)
    # assert model.autoencoder_predict(
    #     test_arr_
    # ), "Failed to make a prediction. Check your network architecture again!"

    ########################
    # Save model and outputs
    ########################
    out_parent = os.path.abspath(out_dir)
    if save_model:
        path = out_parent + "/" + out_model
        model.autoencoder.save(path, overwrite=True, save_format="h5")
        logging.info(f"Model has been saved to {path}")

    if save_weights:
        path = out_parent + "/" + out_weights
        model.autoencoder.save_weights(path, overwrite=True, save_format="h5")
        logging.info(f"Weights of model have been saved to {path}")

    if save_weights_npz:
        filename = os.path.splitext(out_weights_npz)[0]
        path = out_parent + "/" + filename + ".npz"
        savez = dict()
        for i, layer in enumerate(model.autoencoder.layers):
            if layer.get_weights():
                savez["layer" + str(i + 1)] = layer.get_weights()[0]
        np.savez_compressed(path, **savez)
        logging.info(f"Weights of model have been saved to {path}")

    if save_biases_npz:
        filename = os.path.splitext(out_biases_npz)[0]
        path = out_parent + "/" + filename + ".npz"
        savez = dict()
        for i, layer in enumerate(model.autoencoder.layers):
            if layer.get_weights():
                savez["layer" + str(i + 1)] = layer.get_weights()[1]
        np.savez_compressed(path, **savez)
        logging.info(f"Biases of model have been saved to {path}")

    if save_graph:
        model.save_graph(model.autoencoder, model.autoencoder.name, out_dir)
        logging.info(f"Directed graphs of all model have been saved to {os.path.abspath(out_dir)}")

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

        save_path = out_parent + "/" + loss_plot
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

        save_path = out_parent + "/" + metrics_plot
        plt.savefig(save_path)
        logging.info(f"Metric accuracy history plot has been saved to {save_path}")
        if show_metrics:
            fig.show()

    print("=" * 30 + " DONE " + "=" * 30)

    # sys.stdout.close()


if __name__ == "__main__":
    main()

