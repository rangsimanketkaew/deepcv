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
    QGridLayout,
    QFormLayout, QLabel, 
    QLineEdit, 
    QTabWidget, 
    QWidget, 
    QFrame,
    QPushButton, 
    QVBoxLayout
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
        grid.addWidget(self.size_sample, 0, 1, 1, 3)
        grid.addWidget(QCheckBox("Shuffle dataset"), 1, 0)
        grid.addWidget(QCheckBox("Split dataset"), 1, 1)
        grid.addWidget(QLabel("Split ratio:"), 1, 2)
        grid.addWidget(QLineEdit(), 1, 3)
        layout_2.addLayout(grid)
        layout_2.addStretch(1)

        # layout network
        layout_3 = QVBoxLayout()
        grid = QGridLayout()
        # --- widget ---
        grid.addWidget(QLabel("Layer"), 2, 0)
        for i in range(5):
            grid.addWidget(QLabel(f"Hidden {i+1}"), i+3, 0)
            grid.addWidget(QLineEdit(), i+3, 1)
            grid.addWidget(QCheckBox(), i+3, 2)
            grid.addWidget(QCheckBox(), i+3, 3)
            grid.addWidget(QCheckBox(), i+3, 4)
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
        combo_act_func = QComboBox()
        combo_act_func.addItems(["Linear", "Sigmoid", "Tanh", "ReLU", "LeakyReLU"])
        grid.addWidget(QLabel("Hidden layer"), 1, 0)
        grid.addWidget(combo_act_func, 1, 1)
        combo_act_func = QComboBox()
        combo_act_func.addItems(["Linear", "Sigmoid", "Tanh", "ReLU", "LeakyReLU"])
        grid.addWidget(QLabel("Output layer"), 2, 0)
        grid.addWidget(combo_act_func, 2, 1)
        layout_4.addLayout(grid)
        layout_4.addStretch(1)

        # layout techniques
        layout_5 = QVBoxLayout()
        grid = QGridLayout()
        # --- widget ---
        grid.addWidget(QCheckBox("Use Early stopping"))
        grid.addWidget(QCheckBox("Use Reduce learning rate"))
        layout_5.addLayout(grid)
        layout_5.addStretch(1)
        
        # layout output
        layout_6 = QVBoxLayout()
        grid = QGridLayout()
        # --- widget ---
        grid.addWidget(QLabel("Output"), 0, 0)
        grid.addWidget(QCheckBox("Save model (.h5)"), 1, 0)
        grid.addWidget(QCheckBox("Save weight (.npz)"), 1, 1)
        grid.addWidget(QCheckBox("Save bias (.npz)"), 1, 2)
        grid.addWidget(QCheckBox("Save model graph (.png)"), 2, 0)
        grid.addWidget(QCheckBox("Save model summary (.txt)"), 2, 1)
        layout_6.addLayout(grid)
        layout_6.addStretch(1)

        # layout training setting
        layout_7 = QVBoxLayout()
        grid = QGridLayout()
        # --- widget ---
        grid.addWidget(QLabel("Training:"), 0, 0)
        grid.addWidget(QLabel("Epochs:"), 1, 0)
        grid.addWidget(QLineEdit(), 1, 1)
        grid.addWidget(QLabel("Batch size:"), 1, 2)
        grid.addWidget(QLineEdit(), 1, 3)
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
        text = "Input generator of deep learning-based collective variables (DeepCV)\n\
Website: https://gitlab.uzh.ch/lubergroup/deepcv\n\
Version: Development version\n\n\
Support:\n\
- Multi-layer autoencoder\n\
- Non-linear and linear activation functions\n\n\
Author:\n\
- Rangsiman Ketkaew\n\
- Fabrizio Creazzo\n\
- Sandra Luber\n\n\
Developed at Department of Chemistry, University of Zurich, Switzerland"
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
        print("Test preview file", flush=True)
        print(self.size_sample.text())

    def save_file(self):
        print("Test save button", flush=True)

def main():
    app = QApplication([])
    window = Window()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()