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
import numpy as np
from matplotlib import pyplot as plt
from inspect import getmembers, isfunction

from utils import util  # needs to be loaded before calling TF

util.fix_autograph_warning()

import tensorflow as tf

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

from modules import loss


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

    def add_losses(self, _1st_loss, _2nd_loss):
        """Add losses to the class

        Args:
            _1st_loss (func): Main loss
            _2nd_loss (func): Loss-like penalty function
        """
        self.main_loss = _1st_loss
        self.penalty_loss = _2nd_loss

    def generate_layer(self, units, i):
        """Generate input layer for model.

        Args:
            units (int): Number of units/nodes of a layer
            i (int): Index of layer

        Returns:
            Input (obj): Tensorflow input layer
        """
        return Input(shape=(units,), dtype=tf.float32, name="input_layer_" + str(i))

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

        # Input: I1 + I2 + ... = Inp
        self.size_inputs = [i.shape[1] for i in self.train_set]
        self.list_inputs = [
            self.generate_layer(self.size_inputs[i], i + 1) for i in range(len(self.size_inputs))
        ]
        # If there is only one dataset provided, then turn off concatenation, otherwise merge them
        if len(self.list_inputs) == 1:
            self.inputs = self.list_inputs[0]
        else:
            self.inputs = Concatenate(axis=-1)(self.list_inputs)  # Merge branches

        # ---------
        # Encoder: Inp --> H1 --> H2 --> H3
        # ---------
        self.encoded1 = Dense(self.units_1, activation=self.func_1, use_bias=True)(self.inputs)
        self.encoded2 = Dense(self.units_2, activation=self.func_2, use_bias=True)(self.encoded1)
        self.latent = Dense(self.units_3, activation=self.func_3, use_bias=True)(
            self.encoded2
        )  # latent layer

        # ---------
        # Decoder: H3 --> H4 --> H5 --> Out
        # ---------
        self.decoded1 = Dense(self.units_4, activation=self.func_4, use_bias=True)(self.latent)
        self.decoded2 = Dense(self.units_5, activation=self.func_5, use_bias=True)(self.decoded1)
        self.outputs = Dense(self.inputs.shape[1], activation=None, use_bias=True)(
            self.decoded2
        )  # reconstruct input

        # Output: Out --> O1 + O2 + ...
        self.list_outputs = tf.split(
            self.outputs, self.size_inputs, axis=1, name=output_name
        )  # split output into sub-tensors

    def build_encoder(self, name="encoder"):
        """Build encoder model

        Args:
            name (str, optional): Name of model. Defaults to "encoder".
        """
        self.encoder = Model(inputs=self.list_inputs, outputs=self.latent, name=name)

    def build_decoder(self, model_name="decoder", output_name="decoder_output"):
        """Build decoder model

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
            outputs=[tf.split(outputs, self.size_inputs, axis=1, name=output_name)],
            name=model_name,
        )

    def build_autoencoder(self, model_name="daenn"):
        """Build autoencoder model

        Args:
            model_name (str, optional): Name of model. Defaults to "daenn".
        """
        self.autoencoder = Model(inputs=self.list_inputs, outputs=self.list_outputs, name=model_name)

    def parallel_gpu(self, gpus=1):
        """
        Parallelization with multi-GPU

        Args:
            gpus (int): Number of GPUs. Defaults to 1.
        """
        if gpus > 1:
            try:
                from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model

                self.autoencoder = multi_gpu_model(self.autoencoder, gpus=gpus)
                print(">>> Training on multi-GPU on a single machine")
            except:
                print(">>> Warning: cannot enable multi-GPUs support for Keras")

    # def combine_loss(self, y_true, y_pred):
    def combine_loss(self, alpha):
        """Use the closure to make a custom loss be able to receive additional arguments
        But keep in mind that this could yield a potential problem when loading a model

        Args:
            alpha (_type_): _description_

        Returns:
            tensor: Return values from the closure function
        """
        @tf.function
        def _loss(y_true, y_pred):
            return tf.math.reduce_max(tf.subtract(y_true, y_pred)) * alpha

        return _loss

    def compile_model(self, optimizer, loss, loss_weights):
        """Compile neural network model

        Args:
            optimizer (str): Name of optimizer
            loss (str): Name of loss function
            loss_weights (list): Weight for each loss
        """
        self.optimizer = optimizer
        self.loss = loss
        self.loss_weights = loss_weights
        self.autoencoder.compile(
            optimizer=self.optimizer,
            # loss={"tf.split": self.combine_loss(alpha=0.5)},
            loss={"tf.split": "mse"},
            # loss_weights=self.loss_weights,
            metrics=["mse"],
        )

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
        # e.g. 'tensorflow --logdir ./log'
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

    print("=" * 30 + " Program started " + "=" * 30)
    print(f"Project: {project}")

    if neural_network.lower() != "daenn":
        exit(f"Error: This is a DAENN trainer, not {neural_network}.")

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
            print("Warning: LeakyReLU is used as a loss function.")
            return LeakyReLU(alpha=0.2)
        else:
            print(f"Error: Activation function youspecified, {f}, is not supported.")
            print(f"Available functions are {avail_act_func}")
            exit()

    user_act_func = list(map(check_act_func, act_funcs))

    # Loss
    tf_loss = getmembers(tf.keras.losses, isfunction)
    avail_loss = [i[0] for i in tf_loss]
    if main_loss in avail_loss:
        pass
    elif main_loss.lower() == "grmse":
        print("Warning: Customized GRMSE is used as a loss function.")
        main_loss = loss.GRMSE
    else:
        print(f"Error: Loss function you specified, {main_loss}, is not supported.")
        print(f"Available losses are {avail_loss}")
        exit()

    # Penalty loss (loss-like function for DAENN)
    if penalty_loss.lower() == "rmse":
        penalty_loss = loss.RMSE
    elif penalty_loss.lower() == "grmse":
        penalty_loss = loss.GRMSE

    ##############
    # Check output
    ##############
    if not os.path.isdir(out_dir):
        print(
            f"Error: Output directory you specified, {out_dir}, does not exist. Please create this directory!"
        )
        exit()

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
        encoded_test = model.encoder_predict(test_arr)
        decoded_test = model.decoder_predict(encoded_test)

        # Save TF graph
        if save_graph:
            model.save_graph(model.encoder, model.encoder.name, out_dir)
            model.save_graph(model.decoder, model.decoder.name, out_dir)

    ######################################
    # Build, compile and train DAENN model
    ######################################
    model = Autoencoder()
    model.add_dataset(train_arr, test_arr)
    model.build_network(units=units, act_funcs=user_act_func)
    model.build_autoencoder()
    model.add_losses(main_loss, penalty_loss)
    model.compile_model(optimizer, main_loss, loss_weights)
    # show model info
    if show_summary:
        model.autoencoder.summary()
    # Train model
    model.train_model(num_epoch, batch_size, verbosity, save_tb)
    print(">>> Congrats! Training model is completed.")

    # Test prediction
    assert model.autoencoder_predict(
        test_arr
    ), "Failed to make a prediction. Check your network architecture again!"

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
        model.save_graph(model.autoencoder, model.autoencoder.name, out_dir)
        print(f">>> Directed graphs of all model have been saved to {os.path.abspath(out_dir)}")

    # summarize history for loss and accuracy
    if save_loss:
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

    if save_metrics:
        plt.figure(1)
        plt.plot(model.history.history["tf.split_mse"])
        plt.plot(model.history.history["val_tf.split_mse"])
        plt.title("model accuracy")
        plt.ylabel("accuracy metric")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")
        save_path = out_parent + "/" + metrics_plot
        plt.savefig(save_path)
        print(f">>> Metric accuracy history plot has been saved to {save_path}")
        if show_metrics:
            plt.show()

    print("=" * 30 + " DONE " + "=" * 30)


if __name__ == "__main__":
    main()

