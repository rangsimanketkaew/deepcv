"""
Deep Learning for Collective Variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
28/11/2020 : Rangsiman Ketkaew
"""

"""Multi-input neural network
"""

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import argparse
import numpy as np

from utils import util  # needs to be loaded before calling TF
util.tf_logging(2, 3)  # warning level

from helpers import trajectory

from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model, multi_gpu_model
from tensorflow.keras.callbacks import TensorBoard
from scipy.stats import pearsonr, spearmanr, kendalltau


class MultiInputNN:
    def __init__(self):
        super(MultiInputNN, self).__init__()

    def preprocess(self, dataset, ref_mol, training_set_ratio, labels):
        """
        Process data
        """
        # Extract molecules
        self.traj = trajectory.extract_xyz(dataset)
        self.ref_mol = trajectory.extract_xyz(ref_mol)

        # Fitting all frames to reference frame
        # Remove rotational and translational motions
        self.traj_fitted = trajectory.fitting(self.traj, self.ref_mol)

        # Extract labels
        self.num_label = len(labels)
        self.label = np.loadtxt(labels[0], dtype=np.float64, skiprows=1)

        # Shuffle frames
        self.indices = np.arange(self.traj_fitted.shape[0])
        np.random.shuffle(self.indices)
        self.traj_fitted = self.traj_fitted[self.indices]
        self.label = self.label[self.indices]

        # Determine size of training set
        self.training_set_size = int(training_set_ratio * self.traj_fitted.shape[0] / 100)

        # Split dataset
        self.training_set = self.traj_fitted[: self.training_set_size]
        self.testing_set = self.traj_fitted[self.training_set_size :]

        self.training_label = self.label[: self.training_set_size]
        self.testing_label = self.label[self.training_set_size :]

        # -----------------------------------
        # Preprocess the data & Normalization
        # -----------------------------------

        self.traj_size = self.traj_fitted.shape
        self.traj_fitted = self.traj_fitted.reshape(self.traj_size[0], -1)

        max = float(np.max(self.traj_fitted))
        self.training_set_size = self.training_set.shape[0]
        self.testing_set_size = self.testing_set.shape[0]
        self.training_set = (self.training_set.reshape(self.training_set_size, -1).astype("float32") / max)
        self.testing_set = (self.testing_set.reshape(self.testing_set_size, -1).astype("float32") / max)

        self.training_label = self.training_label.astype("float32") / max
        self.testing_label = self.testing_label.astype("float32") / max

    def build_network(self, optimizer, loss, batch_size, num_epoch,
        units_1, units_2, units_3, 
        func_1, func_2, func_3,
    ):
        """
        Multi-input fully-connected neural network
        """
        self.optimizer = optimizer
        self.loss = loss
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.units_1 = units_1
        self.units_2 = units_2
        self.units_3 = units_3
        self.func_1 = func_1
        self.func_2 = func_2
        self.func_3 = func_3

        # define two sets of inputs
        inp_shape = np.prod((self.traj_size[1], self.traj_size[2]))
        self.input_1 = Input(shape=(inp_shape,), name="input_layer_1")  # coordinate
        self.input_2 = Input(shape=(inp_shape,), name="input_layer_2")  # distance

        # the first branch for the first input
        self.branch_1_hidden_1 = Dense(units_1, activation=func_1, use_bias=True)
        self.branch_1_hidden_1 = self.branch_1_hidden_1(self.input_1)
        self.branch_1_hidden_2 = Dense(units_2, activation=func_2, use_bias=True)
        self.branch_1_hidden_2 = self.branch_1_hidden_2(self.branch_1_hidden_1)
        self.branch_1_hidden_3 = Dense(units_3, activation=func_3, use_bias=True)
        self.branch_1 = self.branch_1_hidden_3(self.branch_1_hidden_2)
        self.branch_1_nn = Model(inputs=self.input_1, outputs=self.branch_1, name="branch_1")

        # the second branch for the second input
        self.branch_2_hidden_1 = Dense(units_1, activation=func_1, use_bias=True)
        self.branch_2_hidden_1 = self.branch_2_hidden_1(self.input_2)
        self.branch_2_hidden_2 = Dense(units_2, activation=func_2, use_bias=True)
        self.branch_2_hidden_2 = self.branch_2_hidden_2(self.branch_2_hidden_1)
        self.branch_2_hidden_3 = Dense(units_3, activation=func_3, use_bias=True)
        self.branch_2 = self.branch_2_hidden_3(self.branch_2_hidden_2)
        self.branch_2_nn = Model(inputs=self.input_2, outputs=self.branch_2, name="branch_2")

        self.inputs_combined = Concatenate(axis=-1)(
            [self.branch_1_nn.output, self.branch_2_nn.output]
        )

        self.output_hidden_1 = Dense(2, activation="sigmoid", use_bias=True)
        self.output_hidden_1 = self.output_hidden_1(self.inputs_combined)
        self.output = Dense(1, activation="linear", use_bias=True)
        self.output = self.output(self.output_hidden_1)

    def build_model(self):
        """
        # Build model
        """
        self.inputs = [self.branch_1_nn.input, self.branch_2_nn.input]
        self.nn_model = Model(inputs=self.inputs, outputs=self.output, name="nn_model")

    def parallel_gpu(self, gpus=1):
        """
        Parallelization with multi-GPU
        """
        self.gpus = gpus
        if self.gpus > 1:
            try:
                self.nn_model = multi_gpu_model(self.nn_model, gpus=self.gpus)
                print(">>> Training on multi-GPU on a single machine")
            except:
                print(">>> Warning: cannot enable multi-GPUs support for Keras")

    def compile_model(self):
        """
        Compile model
        """
        self.nn_model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=None,
            loss_weights=None,
            sample_weight_mode=None,
        )

    def show_model(self):
        """
        Show summary of model
        """
        self.nn_model.summary()

    def plot_model_img(self, model_img):
        """
        Plot model and save it as image
        """
        plot_model(
            self.nn_model,
            to_file=model_img,
            show_shapes=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=False,
            dpi=192,
        )

    def train_model(self, verbose=1):
        """
        Train model
        """
        # TF Board
        # You can use tensorboard to visualize TensorFlow runs and graphs.
        # e.g. 'tensorflow --logdir ./log
        log_dir = "./log/"
        tbCallBack = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)

        self.history = self.nn_model.fit(
            x=[self.training_set, self.training_set],  # input
            y=self.training_label,  # label
            shuffle=True,
            validation_split=0.20,
            # validation_data=(testing_set, testing_label),
            epochs=self.num_epoch,
            batch_size=self.batch_size,
            verbose=verbose,
            use_multiprocessing=True,
            callbacks=[tbCallBack],
        )

    def evaluate(self):
        """
        Evaluate model on the test data set
        """
        return self.nn_model.evaluate([self.testing_set, self.testing_set], self.testing_label, batch_size=self.batch_size, verbose=2)

    def predict(self):
        """
        Generate predictions
        """
        # probabilities -- the output of the last layer on new data
        self.encoded_label = self.nn_model.predict(
            [self.traj_fitted, self.traj_fitted]
        )  # shape: (N, 1)

    def calc_corr(self):
        """
        Calculate correlation coefficients
        """
        # Pearson
        self.pearson_all, _ = pearsonr(self.label, self.encoded_label.flatten())
        self.pearson_training, _ = pearsonr(self.training_label, self.encoded_label.flatten()[: self.training_set_size])
        self.pearson_testing, _ = pearsonr(self.testing_label, self.encoded_label.flatten()[self.training_set_size :])
        # Spearman
        self.spearman_all, _ = spearmanr(self.label, self.encoded_label.flatten())
        self.spearman_training, _ = spearmanr(self.training_label, self.encoded_label.flatten()[: self.training_set_size])
        self.spearman_testing, _ = spearmanr(self.testing_label, self.encoded_label.flatten()[self.training_set_size :])
        # Kendall
        self.kendalltau_all, _ = kendalltau(self.label, self.encoded_label.flatten())
        self.kendalltau_training, _ = kendalltau(self.training_label, self.encoded_label.flatten()[: self.training_set_size])
        self.kendalltau_testing, _ = kendalltau(self.testing_label, self.encoded_label.flatten()[self.training_set_size :])


def main():
    info = "Feedforward neural networks with multiple inputs and mixed data."
    parser = argparse.ArgumentParser(description=info)
    parser.add_argument("input", metavar="INPUT", type=str, help="Input file (JSON)")

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        exit('Error: No such file "' + str(args.input) + '"')

    # Load data from JSON input file
    json = util.load_json(args.input)
    project = json["project"]["name"]
    neural_network = json["project"]["neural_network"]
    # ---------
    dataset = json["input"]["dataset"]
    labelset = json["input"]["labelset"]
    ref_mol = json["input"]["ref_mol"]
    # ---------
    out_trained = json["output"]["out_trained"]
    out_model = json["output"]["out_model"]
    out_plumed = json["output"]["out_plumed"]
    model_img = json["output"]["model_img"]
    # ---------
    split = json["dataset"]["split"]
    split_ratio = json["dataset"]["split_ratio"]
    shuffle = json["dataset"]["shuffle"]
    # ---------
    optimizer = json["model"]["optimizer"]
    loss = json["model"]["loss"]
    batch_size = json["model"]["batch_size"]
    num_epoch = json["model"]["num_epoch"]
    # ---------
    num_layers = json["network"]["num_layers"]
    units_1 = json["network"]["units_1"]
    units_2 = json["network"]["units_2"]
    units_3 = json["network"]["units_3"]
    func_1 = json["network"]["func_1"]
    func_2 = json["network"]["func_2"]
    func_3 = json["network"]["func_3"]
    # ---------
    enable_gpu = json["performance"]["enable_gpu"]
    gpus = json["performance"]["gpus"]
    # ---------
    verbosity = json["settings"]["verbosity"]
    save_model = json["settings"]["save_model"]
    save_cv = json["settings"]["save_cv"]
    show_summary = json["settings"]["show_summary"]
    show_layer = json["settings"]["show_layer"]
    show_eva = json["settings"]["show_eva"]
    show_loss = json["settings"]["show_loss"]
    show_corr = json["settings"]["show_corr"]
    show_layer_table = json["settings"]["show_layer_table"]

    # ========================================

    print("=========== Program started ===========")
    print(f"Project: {project}")

    if neural_network.lower() != "multi":
        exit(f"Error: This is a Multi-input NN trainer, not {neural_network}.")

    # Train on GPU?
    if not enable_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    ##################################
    # Build, compile and train a model
    ##################################
    model = MultiInputNN()
    model.preprocess(dataset, ref_mol, split_ratio, labelset)
    model.build_network(
        optimizer, loss, batch_size, num_epoch,
        units_1, units_2, units_3,
        func_1, func_2, func_3,
    )
    model.build_model()
    model.compile_model()
    model.plot_model_img(model_img)
    model.train_model(verbosity)
    model.predict()

    history = model.history
    encoded_label = model.encoded_label

    # save trained model
    if save_model:
        path = os.path.abspath(out_model)
        model.nn_model.save(path, save_format="h5")
        print(f">>> Model saved to {path}")

    if show_summary:
        model.show_model()

    if show_layer:
        print(model.nn_model.layers)

    if show_eva:
        print("===== Evaluate results ======")
        print(f">>> test loss: {model.evaluate()}")

    # summarize history for loss
    if show_loss:
        from matplotlib import pyplot as plt

        plt.figure(1)
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")
        # plt.show(block=False)
        plt.show()

    if show_corr:
        model.calc_corr()
        print("===== Corre coeff =====")
        print(f"Pearson: {model.pearson_all}")
        print(f"Pearson training: {model.pearson_training}")
        print(f"Pearson testing : {model.pearson_testing}")
        print("-----------------------")
        print(f"Spearman: {model.spearman_all}")
        print(f"Spearman training: {model.spearman_training}")
        print(f"Spearman testing : {model.spearman_testing}")
        print("-----------------------")
        print(f"Kendall: {model.kendalltau_all}")
        print(f"Kendall training: {model.kendalltau_training}")
        print(f"Kendall testing : {model.kendalltau_testing}")
        print("-----------------------")

    print("="*30 + " DONE " + "="*30)

if __name__ == "__main__":
    main()

