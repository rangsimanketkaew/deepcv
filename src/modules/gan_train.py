"""
Deep Learning for Collective Variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
28/11/2020 : Rangsiman Ketkaew
"""

"""
Generative adversarial networks (GANs) for generating data from sample noise space
"""

import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import argparse
import time
import numpy as np

from utils import util  # needs to be loaded before calling TF

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten
from tensorflow.keras.layers import LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.utils import plot_model


class GAN_Model(object):
    def __init__(self):
        pass

    def add_dataset(self, input_shape, train_set):
        """Add dataset after creating an instance of Autoencoder class

        Args:
            input_shape (tuple): Shape of input array for each sample
            train_set (list): List containing train sets (NumPy array). The feature of all set must have the same shape.
        """
        self.input_shape = input_shape
        self.train_set = train_set

    def add_batch_norm(self, model, param):
        """Add batch normalization with user-define parameters to a given model

        Args:
            model (class): Model
            param (dict): User-defined parameters for batch normalization: momentum, epsilon, renorm, renorm_momentum
        """
        self.model = model
        momentum, epsilon, renorm, renorm_momentum = list(param.values())
        self.model.add(
            BatchNormalization(
                momentum=momentum, epsilon=epsilon, renorm=renorm, renorm_momentum=renorm_momentum
            )
        )

    def build_generator(
        self,
        noise_shape,
        g_units_1,
        g_units_2,
        g_units_3,
        g_func_1,
        g_func_2,
        g_func_3,
        g_func_4,
        kernel_reg=None,
        g_batch_norm=False,
        g_batch_norm_param=None,
        name="Generator",
    ):
        """Setup hyper-parameters and build Generator model using Sequential API

        Args:
            noise_shape (tuple): Shape of noise input
            g_units_1 (int): Number of units in layer 1 of Generator
            g_units_2 (int): Number of units in layer 2 of Generator
            g_units_3 (int): Number of units in layer 3 of Generator
            g_func_1 (str): Activation function for layer 1 of Generator
            g_func_2 (str): Activation function for layer 2 of Generator
            g_func_3 (str): Activation function for layer 3 of Generator
            g_func_4 (str): Activation function for layer 4 of Generator
            kernel_reg (str): Kernel regularizer for all layers of Generator. Defaults to None.
            g_batch_norm (bool, optional): True if you want to use batch normalization. Defaults to False.
            g_batch_norm_param (dict, optional): Batch normalization parameters. Defaults to None.
            name (str, optional): Name of model. Defaults to "Generator".

        Returns:
            Model (Class): Groups layer of noise
        """
        # 1D array of latent vector / noise
        self.noise_shape = tuple(noise_shape)
        self.g_units_1 = g_units_1
        self.g_units_2 = g_units_2
        self.g_units_3 = g_units_3
        self.g_func_1 = g_func_1
        self.g_func_2 = g_func_2
        self.g_func_3 = g_func_3
        self.g_func_4 = g_func_4
        self.kernel_reg = kernel_reg
        self.g_batch_norm = g_batch_norm
        self.g_batch_norm_param = g_batch_norm_param

        self.G = Sequential(name="Generator")
        # The layer would expect 1D array with noise shape element and it would produce e.g. 256 outputs.
        # Note: Input shape here is a tuple representing how many elements an array or tensor has in each dimension.
        # Layer 1
        self.G.add(
            Dense(
                self.g_units_1,
                input_shape=self.noise_shape,
                name="layer1",
                kernel_regularizer=self.kernel_reg,
            )
        )
        if self.g_func_1.lower() == "leakyrelu":
            self.G.add(LeakyReLU(alpha=0.2))
        if self.g_batch_norm:
            self.add_batch_norm(self.G, self.g_batch_norm_param)
        # Layer 2
        self.G.add(Dense(self.g_units_2, name="layer2", kernel_regularizer=self.kernel_reg))
        if self.g_func_2.lower() == "leakyrelu":
            self.G.add(LeakyReLU(alpha=0.2))
        if self.g_batch_norm:
            self.add_batch_norm(self.G, self.g_batch_norm_param)
        # Layer 3
        self.G.add(Dense(self.g_units_3, name="layer3", kernel_regularizer=self.kernel_reg))
        if self.g_func_3.lower() == "leakyrelu":
            self.G.add(LeakyReLU(alpha=0.2))
        if self.g_batch_norm:
            self.add_batch_norm(self.G, self.g_batch_norm_param)
        # Layer 4
        self.G.add(
            Dense(
                np.prod(self.input_shape),
                activation=self.g_func_4.lower(),
                name="output",
                kernel_regularizer=self.kernel_reg,
            )
        )
        self.G.add(Reshape(self.input_shape))

        noise = Input(shape=self.noise_shape)
        inp = self.G(noise)

        return Model(noise, inp, name=name)

    def build_discriminator(
        self,
        d_units_1,
        d_units_2,
        d_units_3,
        d_func_1,
        d_func_2,
        d_func_3,
        kernel_reg=None,
        d_batch_norm=False,
        d_batch_norm_param=None,
        name="Discriminator",
    ):
        """Setup hyper-parameters and build Discriminator model using Sequential API

        Args:
            d_units_1 (int): Number of units in layer 1 of Discriminator
            d_units_2 (int): Number of units in layer 2 of Discriminator
            d_units_3 (int): Number of units in layer 3 of Discriminator
            d_func_1 (str): Activation function for layer 1 of Discriminator
            d_func_2 (str): Activation function for layer 2 of Discriminator
            d_func_3 (str): Activation function for layer 3 of Discriminator
            kernel_reg (str): Kernel regularizer for all layers of Generator. Defaults to None.
            d_batch_norm (bool, optional): True if you want to use batch normalization. Defaults to False.
            d_batch_norm_param (dict, option): Batch normalization parameters. Defaults to None.
            name (str, optional): Name of model. Defaults to "Discriminator".

        Returns:
            Model (Class): Groups layer of prediction's output
        """

        self.d_units_1 = d_units_1
        self.d_units_2 = d_units_2
        self.d_units_3 = d_units_3
        self.d_func_1 = d_func_1
        self.d_func_2 = d_func_2
        self.d_func_3 = d_func_3
        self.kernel_reg = kernel_reg
        self.d_batch_norm = d_batch_norm
        self.d_batch_norm_param = d_batch_norm_param

        self.D = Sequential(name="Discriminator")
        self.D.add(Flatten(input_shape=self.input_shape))
        ###########
        # Layer 1
        ###########
        self.D.add(Dense(self.d_units_1, name="layer1", kernel_regularizer=self.kernel_reg))
        if self.d_func_1.lower() == "leakyrelu":
            self.D.add(LeakyReLU(alpha=0.2))
        if self.d_batch_norm:
            self.add_batch_norm(self.D, self.d_batch_norm_param)
        ###########
        # Layer 2
        ###########
        self.D.add(Dense(self.d_units_2, name="layer2", kernel_regularizer=self.kernel_reg))
        if self.d_func_2.lower() == "leakyrelu":
            self.D.add(LeakyReLU(alpha=0.2))
        if self.d_batch_norm:
            self.add_batch_norm(self.D, self.d_batch_norm_param)
        ###########
        # Layer 3
        ###########
        if self.d_func_3.lower() == "sigmoid":
            self.D.add(
                Dense(
                    self.d_units_3,
                    activation=self.d_func_3.lower(),
                    name="output",
                    kernel_regularizer=self.kernel_reg,
                )
            )
        if self.d_batch_norm:
            self.add_batch_norm(self.D, self.d_batch_norm_param)

        inp = Input(shape=self.input_shape)
        # The validity is the Discriminator’s guess of input being real or not.
        validity = self.D(inp)

        return Model(inp, validity, name=name)

    def train_gan(self, epochs, batch_size, save_interval):
        """Tran GAN model with given hyperparameters and pre-defined Generator and Discriminator

        Args:
            epochs (int): Number of epochs (training iteration)
            batch_size (int): Batch size
            save_interval (int): Save interval every N step
        """

        self.epochs = epochs
        if str(batch_size) == "sqrt":
            print("Use a common heuristic for batch size: the square root of the size of the dataset")
            self.batch_size = int(np.sqrt(self.train_set.shape[0]))
        else:
            self.batch_size = int(batch_size)
        self.save_interval = save_interval

        self.batch_per_epoch = int(self.train_set.shape[0] / self.batch_size)
        half_batch = int(self.batch_size / 2)
        self.history = {"d_loss": [], "g_loss": []}

        # >>> Train the Discriminator
        for epoch in range(self.epochs):
            for batch in range(self.batch_per_epoch):
                # Select a random half batch of real data
                index = np.random.randint(0, self.train_set.shape[0], half_batch)
                train = self.train_set[index]

                # Create random noise from batch
                noise = np.random.normal(0, 1, (half_batch, self.noise_shape[0]))

                # Generate a half batch of fake data
                gen_inp = generator.predict(noise)

                # Separately train the discriminator on real and fake data
                d_loss_real = discriminator.train_on_batch(train, np.ones((half_batch, 1)))
                d_loss_fake = discriminator.train_on_batch(gen_inp, np.zeros((half_batch, 1)))

                # Take average loss from real and fake data
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # >>> Train the Generator

                # Create new noise vectors as input for generator.
                noise = np.random.normal(0, 1, (self.batch_size, self.noise_shape[0]))

                valid_y = np.array([1] * self.batch_size)

                # Train the generator with noise as x and 1 as y.
                g_loss = gan.train_on_batch(x=noise, y=valid_y)

                self.history["d_loss"].append(d_loss)
                self.history["g_loss"].append(g_loss)

                # Progress bar
                print(f"Epoch: {epoch + 1}/{self.epochs}\tBatch: {batch}/{self.batch_per_epoch}")
                print(f"[D loss: {d_loss[0]:.8f}, acc.: {d_loss[1] * 100:.2f}] : [G loss: {g_loss:.8f}]")

    @staticmethod
    def save_model(model, name):
        """Save model as h5 file

        Args:
            model (class): Model
            name (str): Name of model file (.h5)
        """
        path = os.path.splitext(name)[0]
        model.save(path, save_format="h5")
        print(f">>> Model has been saved to {path}")

    @staticmethod
    def save_graph(model, name):
        """Plot model and save it as image

        Args:
            model (class): File name of model image
            name (str): Name of graph image (.png)
        """

        path = os.path.splitext(name)[0] + ".png"
        plot_model(
            model,
            to_file=path,
            show_shapes=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=False,
            dpi=192,
        )


def main():
    info = "Generative adversarial networks (GANs) for learning latent data from features."
    parser = argparse.ArgumentParser(description=info)
    parser.add_argument(
        "-i",
        "--input",
        dest="input",
        metavar="INPUT",
        type=str,
        required=True,
        help="Input file (JSON) defining configuration, setting, parameters.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        dest="dataset",
        metavar="DATASET",
        type=str,
        required=True,
        nargs="+",
        help="Dataset (train + test sets) for training neural network.",
    )
    parser.add_argument(
        "-k",
        "--key",
        dest="key",
        metavar="KEY",
        type=str,
        required=True,
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
    normalize_scale = json["dataset"]["normalize_scale"]
    max_scale = json["dataset"]["max_scale"]
    # ---------
    optimizer = json["gan"]["optimizer"]
    loss = json["gan"]["loss"]
    regularizer = json["gan"]["regularizer"]
    num_epoch = json["gan"]["num_epoch"]
    batch_size = json["gan"]["batch_size"]
    save_interval = json["gan"]["save_interval"]
    # ---------
    noise_shape = json["generator"]["noise_shape"]
    g_units_1 = json["generator"]["units_1"]
    g_units_2 = json["generator"]["units_2"]
    g_units_3 = json["generator"]["units_3"]
    g_func_1 = json["generator"]["func_1"]
    g_func_2 = json["generator"]["func_2"]
    g_func_3 = json["generator"]["func_3"]
    g_func_4 = json["generator"]["func_4"]
    g_batch_norm = json["generator"]["batch_norm"]
    g_batch_norm_param = json["generator"]["batch_norm_param"]
    # ---------
    d_units_1 = json["discriminator"]["units_1"]
    d_units_2 = json["discriminator"]["units_2"]
    d_units_3 = json["discriminator"]["units_3"]
    d_func_1 = json["discriminator"]["func_1"]
    d_func_2 = json["discriminator"]["func_2"]
    d_func_3 = json["discriminator"]["func_3"]
    d_batch_norm = json["discriminator"]["batch_norm"]
    d_batch_norm_param = json["discriminator"]["batch_norm_param"]
    # ---------
    enable_gpu = json["performance"]["enable_gpu"]
    gpus = json["performance"]["gpus"]
    # ---------
    show_summary = json["settings"]["show_summary"]
    save_graph = json["settings"]["save_graph"]
    save_model = json["settings"]["save_model"]
    show_loss = json["settings"]["show_loss"]
    # ---------
    out_dir = json["output"]["out_dir"]
    out_G_model = json["output"]["out_G_model"]
    out_D_model = json["output"]["out_D_model"]
    out_GAN_model = json["output"]["out_GAN_model"]
    loss_plot = json["output"]["loss_plot"]

    # ========================================

    print("=" * 30 + " Program started " + "=" * 30)
    print(f"Project: {project}")

    if neural_network.lower() != "gan":
        exit(f"Error: This is a GAN trainer, not {neural_network}.")

    ############################
    # Check and prepare datasets
    ############################
    # Extract features (input)
    dataset_arr = [np.load(i)[j] for i, j in zip(args.dataset, args.key)]
    # Use FP32 for speeding training and reducing precision error
    dataset_arr = [i.astype(np.float32) for i in dataset_arr]
    input_shape = (dataset_arr[0].shape[1],)

    print("=== Shape of dataset before merging ===")
    for i, j in enumerate(dataset_arr):
        print(f">>> {i+1}. Dataset: {j.shape}")

    train_set = np.vstack(dataset_arr)

    print("=== Shape of dataset after merging ===")
    print(f">>> Train: {train_set.shape}")

    # Normalize training set
    if float(max_scale) == 0.0:
        try:
            max_scale = np.max(train_set)
        except:
            exit("Error: Cannot determine maximum scale")
    try:
        train_set = (train_set.astype(np.float32) - normalize_scale) / max_scale
    except:
        exit("Error: Normalization failed. Please check scaling parameters")

    # Train on GPU?
    if not enable_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    #####################
    # Prepare regularizer
    #####################
    if not regularizer["name"]:
        regularizer = None
        print("Warning: Regularizer is not used")
    elif regularizer["name"].lower() == "l2":
        regularizer = l1(float(regularizer["factor"]))
    elif regularizer["name"].lower() == "l2":
        regularizer = l2(float(regularizer["factor"]))
    elif regularizer["name"].lower() == "l1_l2":
        regularizer = l1_l2(float(regularizer["factor_1"]), float(regularizer["factor_2"]))
    else:
        exit(
            "Error: Check your regularizer and factor. Set parameter to None if you do not want to apply regularizer."
        )

    #############################
    # Prepare batch normalization
    #############################
    if g_batch_norm:
        g_batch_norm_param = dict(g_batch_norm_param)
    if d_batch_norm:
        d_batch_norm_param = dict(d_batch_norm_param)

    ###################
    # Prepare optimizer
    ###################
    if optimizer["name"].lower() == "adam":
        optimizer = Adam(
            learning_rate=optimizer["learning_rate"], beta_1=optimizer["beta_1"], beta_2=optimizer["beta_2"]
        )
    else:
        exit("Error: Optimizer you specified is not available")

    ######################################
    # Initiate model and load training set
    ######################################
    model = GAN_Model()
    model.add_dataset(input_shape, train_set)

    ############################################
    # Configure, build and compile the generator
    ############################################
    generator = model.build_generator(
        noise_shape,
        g_units_1,
        g_units_2,
        g_units_3,
        g_func_1,
        g_func_2,
        g_func_3,
        g_func_4,
        regularizer,
        g_batch_norm,
        g_batch_norm_param,
    )
    generator.compile(loss=loss, optimizer=optimizer)

    ################################################
    # Configure, build and compile the discriminator
    ################################################
    discriminator = model.build_discriminator(
        d_units_1,
        d_units_2,
        d_units_3,
        d_func_1,
        d_func_2,
        d_func_3,
        regularizer,
        d_batch_norm,
        d_batch_norm_param,
    )
    discriminator.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
    # Feed random input (noise) into Generator
    noise_shape = tuple(noise_shape)
    z = Input(shape=noise_shape)
    fake_inp = generator(z)
    # Note that the Discriminator needs to be stopped while we are training the Generator
    # and Keras does backpropagation automatically
    discriminator.trainable = False
    valid = discriminator(fake_inp)
    # Combine two models (Generator and Discriminator) to build GAN
    # and train only the Generator to fool the Discriminator
    gan = Model(z, valid, name="GAN")
    gan.compile(loss=loss, optimizer=optimizer)

    ####################
    # Start training GAN
    ####################
    start_time = time.time()  # timing
    model.train_gan(epochs=num_epoch, batch_size=batch_size, save_interval=save_interval)
    end_time = time.time()
    print(">>> Training Done!!")
    print(f">>> Elapsed time:  {end_time - start_time:.3f} second")

    # show model info
    if show_summary:
        print("=== Generator (G) ===")
        model.G.summary()
        print("=== Discriminator (D) ===")
        model.D.summary()
        print("=== GAN (combined G and D) ===")
        gan.summary()

    ########################
    # Save model and outputs
    ########################
    out_parent = os.path.abspath(out_dir)
    if save_model:
        model.save_model(model.G, name=f"{out_parent}/{out_G_model}")
        model.save_model(model.D, name=f"{out_parent}/{out_D_model}")
        model.save_model(gan, name=f"{out_parent}/{out_GAN_model}")

    if save_graph:
        model.save_graph(gan, name=f"{out_parent}/{gan.name}")
        model.save_graph(generator, name=f"{out_parent}/{generator.name}")
        model.save_graph(discriminator, name=f"{out_parent}/{discriminator.name}")

    if show_loss:
        from matplotlib import pyplot as plt

        d_loss = np.array(model.history["d_loss"])
        # print(d_loss.shape)
        d_loss_real, d_loss_fake = d_loss.T
        # print(d_loss_real.shape)
        # print(d_loss_fake.shape)
        # from pprint import pprint
        # pprint(model.history["d_loss"])
        # pprint(model.history["g_loss"])
        plt.figure()
        plt.plot(d_loss_real, label="Discriminator loss: Real")
        plt.plot(d_loss_fake, label="Discriminator loss: Fake")
        plt.plot(model.history["g_loss"], label="Generator loss")
        plt.title("Leaky ReLU training")
        plt.ylabel("Loss value")
        plt.xlabel("Epoch")
        plt.legend(loc="upper left")
        save_path = project + "_loss_vs_epoch.png"
        plt.savefig(save_path)
        print(f">>> Loss history plot has been saved to {save_path}")
        plt.show()

    print("=" * 30 + " DONE " + "=" * 30)


if __name__ == "__main__":
    main()

