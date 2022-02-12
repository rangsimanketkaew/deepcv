"""
Deep Learning for Collective Variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
28/11/2020 : Rangsiman Ketkaew
"""

__version__ = "1.0"
__date__ = "October 2020"

from tabulate import tabulate


def main():
    """Welcome message and program description
    """
    print("\n------------------------------------------------")
    print("DeepCV : Deep Learning for Collective Variables")
    print("-------------------------------------------------")
    print(f"version {__version__} : {__date__}")
    print("University of Zurich, Switzerland")
    print("https://gitlab.uzh.ch/lubergroup/deepcv\n")
    print("This version contains the following modules:\n")

    list_of_functions = [
        ["calc_rep.py", "Calculate molecular representation"],
        ["gen_input.py", "Neural network input generator"],
        ["single_train.py", "Single-data fully-connected neural network"],
        ["multi_train.py", "Multi-data fully-connected neural network"],
        ["ae_train.py", "Autoencoder neural network"],
        ["gan_train.py", "Training generative adversarial network (GAN)"],
        ["gan_predict.py", "Generating samples using trained GAN"],
        ["sprint.py", "Calculate SPRINT coordinate"],
        ["deepcv2plumed.py", "Create PLUMED input file"],
        ["deepcv2colvar.py", "Create Colvar input file"],
        ["analysis_FES.py", "FES validation"],
        ["analysis_model.py", "DAENN model analysis and parameters extraction"],
    ]
    t = tabulate(list_of_functions, headers=["Module", "Description"])
    print(t)
    print("\nFor more detail, please review 'README' in the repository.\n")


if __name__ == "__main__":
    main()
