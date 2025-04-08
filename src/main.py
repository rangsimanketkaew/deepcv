#!/usr/bin/env python3

"""
Deep Learning for Collective Variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
23/02/2022 : Rangsiman Ketkaew
"""

__version__ = "1.0"
__date__ = "February 2022"

from tabulate import tabulate

import sys
import argparse


list_of_functions = [
        ["xyz2arr", "Convert xyz to NumPy array"],
        ["stack_array", "Combine arrays of the same dimension"],
        ["calc_rep", "Calculate molecular representations"],
        ["gen_input", "Neural network input generator"],
        ["single_train", "Single-data fully-connected neural network"],
        ["multi_train", "Multi-data fully-connected neural network"],
        ["daenn", "Deep autoencoder neural network"],
        ["gan_train", "Training generative adversarial network (GAN)"],
        ["gan_predict", "Generating samples using trained GAN"],
        ["deepcv2plumed", "Create PLUMED input file"],
        ["analyze_FES", "FES validation"],
        ["analyze_model", "DAENN model analysis and parameters extraction"],
        ["explore_abi", "Calculate exploration ability"],
    ]

def main():
    """Welcome message and program description"""
    print("\n------------------------------------------------")
    print("DeepCV : Deep Learning for Collective Variables")
    print("-------------------------------------------------")
    print(f"version {__version__} : {__date__}")
    print("University of Zurich, Switzerland")
    print("https://gitlab.uzh.ch/lubergroup/deepcv\n")
    t = tabulate(list_of_functions, headers=["Module", "Description"])
    print(t)
    print("\nFor more detail, please review 'README' in the repository.\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="DeepCV API", add_help=False)
    args, rest = parser.parse_known_args()
    rest_arg = []
    rest_arg.extend(rest)
    sys.argv = rest_arg

    if len(sys.argv) == 0:
        main()
        exit()

    modules_list = [i[0] for i in list_of_functions]

    calling = sys.argv[0]
    if not calling in modules_list:
        print(f"'{calling}' is not DeepCV's module. Available modules are: {modules_list}")
        exit()

    import modules, tools, helpers

    if calling == "xyz2arr":
        helpers.xyz2arr.main()
    elif calling == "stack_array":
        helpers.stack_array.main()
    elif calling == "calc_rep":
        tools.calc_rep.main()
    elif calling == "gen_input":
        tools.gen_input.main()
    elif calling == "single_train":
        modules.single_train.main()
    elif calling == "multi_train":
        modules.multi_train.main()
    elif calling == "daenn":
        modules.daenn.main()
    elif calling == "gan_train":
        modules.gan_train.main()
    elif calling == "gan_predict":
        modules.gan_predict.main()
    elif calling == "deepcv2plumed":
        tools.deepcv2plumed.main()
    elif calling == "analyze_FES":
        tools.analyze_FES.main()
    elif calling == "analyze_model":
        tools.analyze_model.main()
    elif calling == "explore_abi":
        tools.explore_abi.main()
