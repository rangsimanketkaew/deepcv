"""
Deep learning-based collective variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
28/11/2020 : Rangsiman Ketkaew
"""

__version__ = "0.1"
__date__ = "October 2020"

from tabulate import tabulate


def main():
    """Welcome message and program description
    """
    print("\n------------------------------------------------")
    print("DeepCV : Deep learning-based collective variables")
    print("-------------------------------------------------")
    print(f"version {__version__} : {__date__}")
    print("University of Zurich, Switzerland")
    print("https://gitlab.uzh.ch/lubergroup/deepcv\n")
    print("This version contains the following modules:\n")

    list_of_functions = [
        ["single_train.py", "Single-data fully-connected neural network"],
        ["multi_train.py", "Multi-data fully-connected neural network"],
        ["gan_train.py", "Training generative adversarial network (GAN)"],
        ["gan_predict.py", "Generating samples using trained GAN"],
        ["ae_train.py", "Autoencoder neural network"],
        ["sprint.py", "Calculate SPRINT coordinate"],
        ["deepcv2plumed.py", "Create PLUMED input file"],
    ]
    t = tabulate(list_of_functions, headers=["Module", "Descriptions"])
    print(t)
    print("\nFor more detail, please review 'README' in the repository.\n")


if __name__ == "__main__":
    main()
