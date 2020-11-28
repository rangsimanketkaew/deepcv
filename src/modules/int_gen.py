"""
Deep learning-based collective variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
28/11/2020 : Rangsiman Ketkaew
"""

"""
Graphical user interface of the deep learning input generator
"""

import sys
from PyQt5.QtWidgets import (
    QApplication, 
    QCheckBox, QComboBox, QFileDialog, 
    QFormLayout, QLabel, 
    QLineEdit, 
    QTabWidget, 
    QWidget, 
    QPushButton, 
    QVBoxLayout
)


class Window(QWidget):
    def __init__(self):
        super(Window, self).__init__()
        self.setWindowTitle("DeepCV input generator")
        self.resize(800, 500)
        layout = QVBoxLayout()
        # Menu bar
        tabs = QTabWidget()
        tabs.addTab(self.autoencoderTabUI(), "Autoencoder")
        tabs.addTab(self.infoTabUI(), "Info")
        layout.addWidget(tabs)
        layout.addStretch()
        self.setLayout(layout)

    def autoencoderTabUI(self):
        networkTab = QWidget()
        layout = QVBoxLayout()

        layout_form = QFormLayout()
        layout_form.addRow("Size of sample (features:)", QLineEdit())
        layout_form.addRow("Split ratio:", QLineEdit())
        layout_form.addRow("Number of neurons in layer 1:", QLineEdit())
        layout_form.addRow("Number of neurons in layer 2:", QLineEdit())
        layout_form.addRow("Number of neurons in layer 3:", QLineEdit())
        layout_form.addRow("Number of neurons in layer 4:", QLineEdit())
        layout_form.addRow("Number of neurons in layer 5:", QLineEdit())
        layout_form.addRow("Number of neurons in output layer:", QLineEdit())

        layout_act_func = QVBoxLayout()
        label_act_func = QLabel("Activation function:")
        layout_act_func.addWidget(label_act_func)

        layout_combo = QVBoxLayout()
        combo_act_func = QComboBox()
        combo_act_func.addItems(["Linear", "Sigmoid", "Tanh", "ReLU", "LeakyReLU"])
        layout_combo.addWidget(combo_act_func)

        layout_check_techniques = QVBoxLayout()
        layout_check_techniques.addWidget(QCheckBox("Use DropOut"))
        layout_check_techniques.addWidget(QCheckBox("Use Regularizer"))
        layout_check_techniques.addWidget(QCheckBox("Use Batch normalization"))
        layout_check_techniques.addWidget(QCheckBox("Use Early stopping"))
        layout_check_techniques.addWidget(QCheckBox("Use Reduce learning rate"))
        
        # input
        layout_box_input = QVBoxLayout()
        button_loadfile = QPushButton("Load dataset")
        button_loadfile.pressed.connect(self.loadFile)
        layout_box_input.addWidget(button_loadfile)

        # output
        layout_output = QVBoxLayout()
        label_output = QLabel("Output:")
        layout_output.addWidget(label_output)
        layout_check_output = QVBoxLayout()
        layout_check_output.addWidget(QCheckBox("Save model (.h5)"))
        layout_check_output.addWidget(QCheckBox("Save weight (.npz)"))
        layout_check_output.addWidget(QCheckBox("Save bias (.npz)"))
        layout_check_output.addWidget(QCheckBox("Save model graph (.png)"))
        layout_check_output.addWidget(QCheckBox("Save model summary (.txt)"))

        # training settings
        layout_train = QVBoxLayout()
        label_train = QLabel("Training:")
        layout_train.addWidget(label_train)
        layout_train_settings = QFormLayout()
        layout_train_settings.addRow("Epochs:", QLineEdit())
        layout_train_settings.addRow("Batch size:", QLineEdit())

        # generate file
        layout_save = QVBoxLayout()
        button_preview = QPushButton("Preview Input")
        button_preview.pressed.connect(self.preview_input)
        button_save = QPushButton("Save")
        button_save.pressed.connect(self.save_file)
        layout_save.addWidget(button_preview)
        layout_save.addWidget(button_save)

        # Combine layouts
        layout.addLayout(layout_box_input)
        layout.addLayout(layout_form)
        layout.addLayout(layout_act_func)
        layout.addLayout(layout_combo)
        layout.addLayout(layout_check_techniques)
        layout.addLayout(layout_train)
        layout.addLayout(layout_train_settings)
        layout.addLayout(layout_output)
        layout.addLayout(layout_check_output)
        layout.addLayout(layout_save)

        networkTab.setLayout(layout)
        return networkTab

    def infoTabUI(self):
        infoTab = QWidget()
        layout = QVBoxLayout()
        text = "Input generator of deep learning-based collective variables (DeepCV)\n\
Website: https://gitlab.uzh.ch/lubergroup/deepcv\n\
Version: Development version\n\n\
Support:\n\
- Multi-layer autoencoder\n\
- Non-linear and linear activation functions\n\n\
Author:\n\
- Rangsiman Ketkaew - University of Zurich"
        label_author = QLabel(text)
        layout.addWidget(label_author)
        infoTab.setLayout(layout)
        return infoTab

    def loadFile(self):
        filter = "TXT (*.txt);;PDF (*.pdf)"
        file_name = QFileDialog()
        file_name.setFileMode(QFileDialog.ExistingFiles)
        names = file_name.getOpenFileNames(self, "Open files")
        print(names[:-1])

    def preview_input(self):
        pass

    def save_file(self):
        print("Test save button")

def main():
    app = QApplication([])
    window = Window()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()