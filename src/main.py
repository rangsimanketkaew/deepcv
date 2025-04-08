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


def main():
    """Welcome message and program description"""
    print("\n------------------------------------------------")
    print("DeepCV : Deep Learning for Collective Variables")
    print("-------------------------------------------------")
    print(f"version {__version__} : {__date__}")
    print("University of Zurich, Switzerland")
    print("https://gitlab.uzh.ch/lubergroup/deepcv\n")

    list_of_functions = [
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

    calling = sys.argv[0]
    modules_list = [
        "calc_rep",
        "gen_input",
        "single_train",
        "multi_train",
        "daenn",
        "gan_train",
        "gan_predict",
        "deepcv2plumed",
        "analyze_FES",
        "analyze_model",
        "explore_abi",
    ]
    if not calling in modules_list:
        print(f"'{calling}' is not DeepCV's module. Available modules are: {modules_list}")
        exit()

    import modules, tools

    if calling == "calc_rep":
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
