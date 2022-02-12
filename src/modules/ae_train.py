"""
Deep Learning for Collective Variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
28/11/2020 : Rangsiman Ketkaew
"""

"""
Autoencoder for learning collective variables from features
"""

import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import argparse
import numpy as np
from inspect import getmembers, isfunction

from utils import util  # needs to be loaded before calling TF

util.tf_logging(2, 3)  # warning level
util.limit_gpu_growth()
util.fix_autograph_warning()

from modules.loss import GRMSE

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split


class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()

    def add_dataset(self, train_set, test_set):
        """Add dataset after creating an instance of Autoencoder class

        Args:
            train_set (list): List containing train sets (NumPy array). The feature of all set must have the same shape.
            test_set (list): List containing test sets (NumPy array). The feature of all set must have the same shape.
        """
        self.train_set = train_set
        self.test_set = test_set

    def generate_layer(self, units, i):
        """Generate input layer for model.

        Args:
            units (int): Number of units/nodes of a layer
            i (int): Index of layer

        Returns:
            Input: Tensorflow input layer
        """
        return Input(shape=(units,), dtype=tf.float32, name="input_layer_" + str(i))

    def build_network(
        self, units_1, units_2, units_3, units_4, units_5, func_1, func_2, func_3, func_4, func_5
    ):
        """Multiple input fully-connected feedforward neural network. 
        This network comprises input layer(s), 5 hidden layers, and output layer(s).

        Args:
            units_1 (int): Number of neurons of layer 1
            units_2 (int): Number of neurons of layer 2
            units_3 (int): Number of neurons of layer 3
            units_4 (int): Number of neurons of layer 4
            units_5 (int): Number of neurons of layer 5
            func_1 (str): Activation function for layer 1
            func_2 (str): Activation function for layer 2
            func_3 (str): Activation function for layer 3
            func_4 (str): Activation function for layer 4
            func_5 (str): Activation function for layer 5
        """
        self.units_1, self.units_2, self.units_3, self.units_4, self.units_5 = (
            units_1,
            units_2,
            units_3,
            units_4,
            units_5,
        )
        self.func_1, self.func_2, self.func_3, self.func_4, self.func_5 = (
            func_1,
            func_2,
            func_3,
            func_4,
            func_5,
        )

        # Input: I1 + I2 + ... = Inp
        self.size_inputs = [i.shape[1] for i in self.train_set]
        self.list_inputs = [
            self.generate_layer(self.size_inputs[i], i + 1) for i in range(len(self.size_inputs))
        ]
        # Check there is only one dataset provided, turn off concatenation
        if len(self.list_inputs) == 1:
            self.inputs = self.list_inputs[0]
        else:
            self.inputs = Concatenate(axis=-1)(self.list_inputs)  # Merge branches

        # Encoder: Inp --> H1 --> H2 --> H3
        self.encoded1 = Dense(self.units_1, activation=self.func_1, use_bias=True)(self.inputs)
        self.encoded2 = Dense(self.units_2, activation=self.func_2, use_bias=True)(self.encoded1)
        self.latent = Dense(self.units_3, activation=self.func_3, use_bias=True)(
            self.encoded2
        )  # latent layer

        # Decoder: H3 --> H4 --> Out
        self.decoded1 = Dense(self.units_4, activation=self.func_4, use_bias=True)(self.latent)
        self.decoded2 = Dense(self.units_5, activation=self.func_5, use_bias=True)(self.decoded1)
        self.outputs = Dense(self.inputs.shape[1], activation=None, use_bias=True)(
            self.decoded2
        )  # reconstruct input

        # Output: Out --> O1 + O2 + ...
        self.list_outputs = tf.split(self.outputs, self.size_inputs, axis=1)  # split output into sub-tensors

    def build_encoder(self, name="encoder"):
        """Build encoder model

        Args:
            name (str, optional): Name of model. Defaults to "encoder".
        """
        self.encoder = Model(inputs=self.list_inputs, outputs=self.latent, name=name)

    def build_decoder(self, name="decoder"):
        """Build decoder model

        Args:
            name (str, optional): Name of model. Defaults to "decoder".
        """
        # We can't use a latent (Tensor) dense layer as an input layer of the decoder because
        # the input of Model class accepts only Input layer. We therefore need to re-construct
        # the decoder based on the second half of Autoencoder.
        decoder_input = Input(shape=(self.latent.shape[1],), name="decoder_input")
        decoded1 = Dense(self.units_4, activation=self.func_4, use_bias=True)(decoder_input)
        decoded2 = Dense(self.units_5, activation=self.func_5, use_bias=True)(decoded1)
        outputs = Dense(self.inputs.shape[1], activation=None, use_bias=True)(decoded2)  # reconstruct input
        self.decoder = Model(
            inputs=decoder_input, outputs=[tf.split(outputs, self.size_inputs, axis=1)], name=name
        )

    def build_autoencoder(self, name="autoencoder"):
        """Build autoencoder model

        Args:
            name (str, optional): Name of model. Defaults to "autoencoder".
        """
        self.autoencoder = Model(inputs=self.list_inputs, outputs=self.list_outputs, name=name)

    def parallel_gpu(self, gpus=1):
        """
        Parallelization with multi-GPU

        Args:
            gpus (int): Number of GPUs. Defaults to 1.
        """
        if gpus > 1:
            try:
                from tensorflow.keras.utils import multi_gpu_model

                self.autoencoder = multi_gpu_model(self.autoencoder, gpus=gpus)
                print(">>> Training on multi-GPU on a single machine")
            except:
                print(">>> Warning: cannot enable multi-GPUs support for Keras")

    def compile_model(self, optimizer, loss):
        """Compile neural network model

        Args:
            optimizer (str): Name of optimizer
            loss (str): Name of loss function
        """
        self.optimizer = optimizer
        self.loss = loss
        self.autoencoder.compile(optimizer=self.optimizer, loss=self.loss, metrics=None)

    def train_model(self, num_epoch, batch_size, verbose=1, log_dir="./logs/"):
        """Train model
        
        Args:
            num_epoch (int): Number of epochs
            batch_size (int): Batch size
        """
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        if self.batch_size == 0:
            self.batch_size = None

        # TF Board
        # You can use tensorboard to visualize TensorFlow runs and graphs.
        # e.g. 'tensorflow --logdir ./log
        tbCallBack = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)

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
            # callbacks=[tbCallBack],
        )

    def encoder_predict(self, input_sample):
        """Generate predictions"""
        return self.encoder.predict(input_sample)

    def decoder_predict(self, encoded_sample):
        """Generate predictions"""
        return self.decoder.predict(encoded_sample)

    def autoencoder_predict(self, test_set):
        """Generate predictions"""
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
    parser.add_argument(
        "-d",
        "--dataset",
        metavar="DATASET",
        type=str,
        required=True,
        nargs="+",
        help="Dataset (train + test sets) for training neural network.",
    )
    parser.add_argument(
        "-k",
        "--key",
        metavar="KEY",
        type=str,
        nargs="+",
        help="Keyword name (dictionary key) of the dataset array in NumPy's compressed file. \
            The number of keyword must consistent with that of npz files.",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        exit('Error: No such file "' + str(args.input) + '"')

    # Load data from JSON input file
    json = util.load_json(args.input)
    project = json["project"]["name"]
    neural_network = json["project"]["neural_network"]
    # ---------
    split = json["dataset"]["split"]
    split_ratio = json["dataset"]["split_ratio"]
    shuffle = json["dataset"]["shuffle"]
    normalize_scale = json["dataset"]["normalize_scale"]
    max_scale = json["dataset"]["max_scale"]
    # ---------
    optimizer = json["model"]["optimizer"]
    loss = json["model"]["loss"]
    batch_size = json["model"]["batch_size"]
    num_epoch = json["model"]["num_epoch"]
    # ---------
    hidden_layers = json["network"]["hidden_layers"]
    units_1 = json["network"]["units_1"]
    units_2 = json["network"]["units_2"]
    units_3 = json["network"]["units_3"]
    units_4 = json["network"]["units_4"]
    units_5 = json["network"]["units_5"]
    func_1 = json["network"]["func_1"]
    func_2 = json["network"]["func_2"]
    func_3 = json["network"]["func_3"]
    func_4 = json["network"]["func_4"]
    func_5 = json["network"]["func_5"]
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
    # ---------
    out_dir = json["output"]["out_dir"]
    out_model = json["output"]["out_model"]
    out_weights = json["output"]["out_weights"]
    out_weights_npz = json["output"]["out_weights_npz"]
    out_biases_npz = json["output"]["out_biases_npz"]
    loss_plot = json["output"]["loss_plot"]

    # ========================================

    print("=" * 30 + " Program started " + "=" * 30)
    print(f"Project: {project}")

    if neural_network.lower() != "ae":
        exit(f"Error: This is an Autoencoder trainer, not {neural_network}.")

    ############################
    # Check and prepare datasets
    ############################
    # Extract features (input)
    if not args.key:
        print("Warning: No npz keys specified, the first key found in array.files is automatically used.")
        dataset_arr_raw = [np.load(i) for i in args.dataset]
        dataset_arr = [i[i.files[0]] for i in dataset_arr_raw]
    else:
        dataset_arr = [np.load(i)[j] for i, j in zip(args.dataset, args.key)]

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
    # Activation function
    tf_act_func = getmembers(tf.keras.activations, isfunction)
    avail_act_func = [i[0] for i in tf_act_func]

    def check_act_func(f):
        if f in avail_act_func:
            return f
        if f.lower() == "leakyrelu":
            print("Warning: LeakyReLU is used as a loss function.")
            return LeakyReLU(alpha=0.2)
        else:
            print(f"Error: Activation function youspecified, {f}, is not supported.")
            print(f"Available functions are {avail_act_func}")
            exit()

    user_act_func = [func_1, func_2, func_3, func_4, func_5]
    user_act_func = list(map(check_act_func, user_act_func))

    # Loss
    tf_loss = getmembers(tf.keras.losses, isfunction)
    avail_loss = [i[0] for i in tf_loss]
    if loss in avail_loss:
        pass
    elif loss.lower() == "grmse":
        print("Warning: Customized GRMSE is used as a loss function.")
        loss = GRMSE
    else:
        print(f"Error: Loss function you specified, {loss}, is not supported.")
        print(f"Available losses are {avail_loss}")
        exit()

    ##############
    # Check output
    ##############
    if not os.path.isdir(out_dir):
        print(
            f"Error: Output directory you specified, {out_dir}, does not exist. Please create this directory!"
        )
        exit()

    ################################
    # Build, compile and train model
    ################################
    model = Autoencoder()
    model.add_dataset(train_arr, test_arr)
    model.build_network(units_1, units_2, units_3, units_4, units_5, *user_act_func)
    model.build_encoder()
    model.build_decoder()
    model.build_autoencoder()
    model.compile_model(optimizer, loss)
    # show model info
    if show_summary:
        model.encoder.summary()
        model.decoder.summary()
        model.autoencoder.summary()
    # Train model
    model.train_model(num_epoch, batch_size, verbosity, save_tb)
    print(">>> Congrats! Training model is completed.")

    # Prediction
    encoded_test = model.encoder_predict(test_arr)
    decoded_test = model.decoder_predict(encoded_test)

    ########################
    # Save model and outputs
    ########################
    out_parent = os.path.abspath(out_dir)
    if save_model:
        path = out_parent + "/" + out_model
        model.autoencoder.save(path, overwrite=True, save_format="h5")
        print(f">>> Model has been saved to {path}")

    if save_weights:
        path = out_parent + "/" + out_weights
        model.autoencoder.save_weights(path, overwrite=True, save_format="h5")
        print(f">>> Weights of model have been saved to {path}")

    if save_weights_npz:
        filename = os.path.splitext(out_weights_npz)[0]
        path = out_parent + "/" + filename + ".npz"
        savez = dict()
        for i, layer in enumerate(model.autoencoder.layers):
            if layer.get_weights():
                savez["layer" + str(i + 1)] = layer.get_weights()[0]
        np.savez_compressed(path, **savez)
        print(f">>> Weights of model have been saved to {path}")

    if save_biases_npz:
        filename = os.path.splitext(out_biases_npz)[0]
        path = out_parent + "/" + filename + ".npz"
        savez = dict()
        for i, layer in enumerate(model.autoencoder.layers):
            if layer.get_weights():
                savez["layer" + str(i + 1)] = layer.get_weights()[1]
        np.savez_compressed(path, **savez)
        print(f">>> Biases of model have been saved to {path}")

    if save_graph:
        model.save_graph(model.encoder, model.encoder.name, out_dir)
        model.save_graph(model.decoder, model.decoder.name, out_dir)
        model.save_graph(model.autoencoder, model.autoencoder.name, out_dir)
        print(f">>> Directed graphs of all model have been saved to {os.path.abspath(out_dir)}")

    # summarize history for loss
    if save_loss:
        from matplotlib import pyplot as plt

        plt.figure(1)
        plt.plot(model.history.history["loss"])
        plt.plot(model.history.history["val_loss"])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")
        save_path = out_parent + "/" + loss_plot
        plt.savefig(save_path)
        print(f">>> Loss history plot has been saved to {save_path}")
        if show_loss:
            plt.show()

    print("=" * 30 + " DONE " + "=" * 30)


if __name__ == "__main__":
    main()

