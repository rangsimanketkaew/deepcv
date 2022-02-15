"""
Deep Learning for Collective Variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
28/11/2020 : Rangsiman Ketkaew
"""

"""
Graphical user interface of the deep learning input generator
"""

import sys
import json
from PyQt5 import QtCore
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QLabel,
    QLineEdit,
    QTabWidget,
    QWidget,
    QFrame,
    QPushButton,
    QVBoxLayout,
)


class Window(QWidget):
    def __init__(self):
        super(Window, self).__init__()
        self.setWindowTitle("DeepCV Input Generator")
        self.resize(400, 500)
        layout = QVBoxLayout()
        # Menu bar
        tabs = QTabWidget()
        tabs.addTab(self.autoencoderTabUI(), "Autoencoder")
        tabs.addTab(self.infoTabUI(), "Info")
        layout.addWidget(tabs)
        layout.addStretch()
        self.setLayout(layout)

    def autoencoderTabUI(self):
        networkTab = QFrame()
        # networkTab.setStyleSheet("margin:5px; border:1px solid")

        # layout input
        layout_1 = QVBoxLayout()
        grid = QGridLayout()
        # --- widget ---
        grid.addWidget(QLabel("Load dataset"), 0, 0)
        button_loadfile = QPushButton("Dataset 1")
        button_loadfile.pressed.connect(self.loadFile)
        grid.addWidget(button_loadfile, 1, 0)
        button_loadfile = QPushButton("Dataset 2")
        button_loadfile.pressed.connect(self.loadFile)
        grid.addWidget(button_loadfile, 1, 1)
        button_loadfile = QPushButton("Dataset 3")
        button_loadfile.pressed.connect(self.loadFile)
        grid.addWidget(button_loadfile, 1, 2)
        layout_1.addLayout(grid)
        layout_1.addStretch(1)

        # layout dataset
        layout_2 = QVBoxLayout()
        grid = QGridLayout()
        # --- widget ---
        grid.addWidget(QLabel("Size of sample (features):"), 0, 0)
        self.size_sample = QLineEdit()
        self.size_sample.setAlignment(QtCore.Qt.AlignCenter)
        grid.addWidget(self.size_sample, 0, 1, 1, 3)
        self.shuffle_dataset = QCheckBox("Shuffle dataset")
        grid.addWidget(self.shuffle_dataset, 1, 0)
        self.split_dataset = QCheckBox("Split dataset")
        self.split_dataset.setChecked(True)
        grid.addWidget(self.split_dataset, 1, 1)
        grid.addWidget(QLabel("Split ratio:"), 1, 2)
        self.split_ratio = QLineEdit("0.8")
        self.split_ratio.setAlignment(QtCore.Qt.AlignCenter)
        grid.addWidget(self.split_ratio, 1, 3)
        layout_2.addLayout(grid)
        layout_2.addStretch(1)

        # layout network
        layout_3 = QVBoxLayout()
        grid = QGridLayout()
        # --- widget ---
        grid.addWidget(QLabel("Layer"), 2, 0)
        default_neuron = [32, 8, 2, 8, 32]
        self.list_num_neuron = []
        for i in range(5):
            grid.addWidget(QLabel(f"Hidden {i+1}"), i + 3, 0)
            num_neuron = QLineEdit(str(default_neuron[i]))
            num_neuron.setAlignment(QtCore.Qt.AlignCenter)
            grid.addWidget(num_neuron, i + 3, 1)
            self.list_num_neuron.append(num_neuron)
            grid.addWidget(QCheckBox(), i + 3, 2)
            grid.addWidget(QCheckBox(), i + 3, 3)
            grid.addWidget(QCheckBox(), i + 3, 4)
        grid.addWidget(QLabel("Neurons"), 2, 1)
        grid.addWidget(QLabel("Dropout"), 2, 2)
        grid.addWidget(QLabel("Batch norm"), 2, 3)
        grid.addWidget(QLabel("Regularization"), 2, 4)
        layout_3.addLayout(grid)
        layout_3.addStretch(1)

        # layout activation function
        layout_4 = QVBoxLayout()
        grid = QGridLayout()
        # --- widget ---
        grid.addWidget(QLabel("Activation function"), 0, 0)
        list_act_func = ["Linear", "Sigmoid", "Tanh", "ReLU", "LeakyReLU"]
        self.combo_act_func_hidden = QComboBox()
        self.combo_act_func_hidden.addItems(list_act_func)
        grid.addWidget(QLabel("Hidden layer"), 1, 0)
        grid.addWidget(self.combo_act_func_hidden, 1, 1)
        self.combo_act_func_output = QComboBox()
        self.combo_act_func_output.addItems(list_act_func)
        grid.addWidget(QLabel("Output layer"), 2, 0)
        grid.addWidget(self.combo_act_func_output, 2, 1)
        layout_4.addLayout(grid)
        layout_4.addStretch(1)

        # layout techniques
        layout_5 = QVBoxLayout()
        grid = QGridLayout()
        # --- widget ---
        self.early_stopping = QCheckBox("Use Early stopping")
        grid.addWidget(self.early_stopping)
        self.reduce_learning_rate = QCheckBox("Use Reduce learning rate")
        grid.addWidget(self.reduce_learning_rate)
        layout_5.addLayout(grid)
        layout_5.addStretch(1)

        # layout output
        layout_6 = QVBoxLayout()
        grid = QGridLayout()
        # --- widget ---
        grid.addWidget(QLabel("Output"), 0, 0)
        self.save_model = QCheckBox("Save model (.h5)")
        self.save_model.setChecked(True)
        grid.addWidget(self.save_model, 1, 0)
        self.save_weight = QCheckBox("Save weight (.npz)")
        self.save_weight.setChecked(True)
        grid.addWidget(self.save_weight, 1, 1)
        self.save_bias = QCheckBox("Save bias (.npz)")
        self.save_bias.setChecked(True)
        grid.addWidget(self.save_bias, 1, 2)
        self.save_model_graph = QCheckBox("Save model graph (.png)")
        self.save_model_graph.setChecked(True)
        grid.addWidget(self.save_model_graph, 2, 0)
        self.save_model_summary = QCheckBox("Save model summary (.txt)")
        self.save_model_summary.setChecked(True)
        grid.addWidget(self.save_model_summary, 2, 1)
        layout_6.addLayout(grid)
        layout_6.addStretch(1)

        # layout training setting
        layout_7 = QVBoxLayout()
        grid = QGridLayout()
        # --- widget ---
        grid.addWidget(QLabel("Training:"), 0, 0)
        grid.addWidget(QLabel("Epochs:"), 1, 0)
        self.epochs = QLineEdit("300")
        self.epochs.setAlignment(QtCore.Qt.AlignCenter)
        grid.addWidget(self.epochs, 1, 1)
        grid.addWidget(QLabel("Batch size:"), 1, 2)
        self.batch_size = QLineEdit("64")
        self.batch_size.setAlignment(QtCore.Qt.AlignCenter)
        grid.addWidget(self.batch_size, 1, 3)
        layout_7.addLayout(grid)
        layout_7.addStretch(1)

        # layout generate file
        layout_8 = QVBoxLayout()
        grid = QGridLayout()
        # --- widget ---
        button_preview = QPushButton("Preview Input")
        button_preview.pressed.connect(self.preview_input)
        button_save = QPushButton("Save")
        button_save.pressed.connect(self.save_file)
        grid.addWidget(button_preview, 0, 0)
        grid.addWidget(button_save, 0, 1)
        layout_8.addLayout(grid)
        layout_8.addStretch(1)

        # Combine layouts
        layout = QVBoxLayout()
        layout.addLayout(layout_1)
        layout.addLayout(layout_2)
        layout.addLayout(layout_3)
        layout.addLayout(layout_4)
        layout.addLayout(layout_5)
        layout.addLayout(layout_6)
        layout.addLayout(layout_7)
        layout.addLayout(layout_8)
        networkTab.setLayout(layout)
        return networkTab

    def infoTabUI(self):
        infoTab = QWidget()
        layout = QVBoxLayout()
        text = "Input generator of Deep Learning for Collective Variables (DeepCV)\n\
Source code: https://gitlab.uzh.ch/lubergroup/deepcv\n\
Version: Development version\n\n\
Support:\n\
- Multi-layer autoencoder\n\
- Non-linear and linear activation functions\n\
- Loss-like penalty functions\n\n\
Author:\n\
- Rangsiman Ketkaew\n\
- Sandra Luber\n\n\
Developed at Department of Chemistry, University of Zurich, Switzerland"
        label_author = QLabel(text)
        layout.addWidget(label_author)
        infoTab.setLayout(layout)
        return infoTab

    def loadFile(self):
        filter = "NPZ (*.npz);;All files (*)"
        file_name = QFileDialog()
        file_name.setFileMode(QFileDialog.ExistingFiles)
        names = file_name.getOpenFileNames(self, "Open files", "", filter)
        print(f"Loaded files: {names[:-1]}")

    def update_data(self):
        self.data = {
            "_comment": "Configuration input file",
            "project": {
                "_comment": "Type of neural network to train",
                "name": "Generated by DeepCV input generator",
                "neural_network": "ae",
            },
            "dataset": {
                "_comment": "Dataset manipulation: dataset splitting and normalization",
                "split": self.split_dataset.isChecked(),
                "split_ratio": float(self.split_ratio.text()),
                "shuffle": self.shuffle_dataset.isChecked(),
                "normalize_scale": 0.0,
                "max_scale": 2,
            },
            "model": {
                "_comment": "Define the optimizer, loss function, number of epochs, and batch size",
                "optimizer": "adam",
                "loss": "mean_absolute_error",
                "num_epoch": int(self.epochs.text()),
                "batch_size": int(self.batch_size.text()),
            },
            "network": {
                "_comment": "Number of neurons and activation function for each hidden layer",
                "hidden_layers": 5,
                "units_1": int(self.list_num_neuron[0].text()),
                "units_2": int(self.list_num_neuron[1].text()),
                "units_3": int(self.list_num_neuron[2].text()),
                "units_4": int(self.list_num_neuron[3].text()),
                "units_5": int(self.list_num_neuron[4].text()),
                "func_1": self.combo_act_func_hidden.currentText(),
                "func_2": self.combo_act_func_hidden.currentText(),
                "func_3": self.combo_act_func_hidden.currentText(),
                "func_4": self.combo_act_func_hidden.currentText(),
                "func_5": self.combo_act_func_hidden.currentText(),
            },
            "performance": {
                "_comment": "Setting for training performance",
                "enable_gpu": True,
                "gpus": 1,
            },
            "settings": {
                "_comment": "User-defined settings",
                "verbosity": 1,
                "show_summary": True,
                "save_tb": False,
                "save_model": self.save_model.isChecked(),
                "save_weights": self.save_weight.isChecked(),
                "save_weights_npz": self.save_weight.isChecked(),
                "save_biases_npz": self.save_bias.isChecked(),
                "save_graph": self.save_model_graph.isChecked(),
                "show_loss": True,
            },
            "output": {
                "_comment": "Set name of output files",
                "out_dir": "output",
                "out_model": "model.h5",
                "out_weights": "model_weights.h5",
                "out_weights_npz": "model_weights.npz",
                "out_biases_npz": "model_biases.npz",
                "loss_plot": "loss.png",
            },
        }

    def preview_input(self):
        print("Preview input file", flush=True)
        print("==================")
        self.update_data()
        print(json.dumps(self.data, indent=4, separators=(",", ": ")))

    def save_file(self):
        name = QFileDialog.getSaveFileName(self, "Save file", "", "JSON (*.json)")
        try:
            with open(name[0], "w") as f:
                json.dump(self.data, f, indent=4, separators=(",", ": "))
            print(f"Input file has been saved to {name[0]}")
        except FileNotFoundError:
            print("Failed to save file. Try again.")


def main():
    app = QApplication([])
    window = Window()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
