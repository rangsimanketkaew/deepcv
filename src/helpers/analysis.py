"""
Deep Learning for Collective Variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
28/11/2020 : Rangsiman Ketkaew
"""


import argparse
from tensorflow.keras import models
import numpy as np


if __name__ == "__main__":
    info = "Check, load, and show trained Model's outputs."
    parser = argparse.ArgumentParser(description=info)
    parser.add_argument(
        "-i", "--input", metavar="INPUT", dest="file", type=str, required=True,
        help="Path to directory where saved model is stored, or a trained weight file, a trained bias file, an output file from DeepCV.")
    parser.add_argument(
        "-t", "--type", metavar="TYPE", dest="type", type=str, required=True,
        choices=["model", "weight", "bias", "deepcv"],
        help="Type of file. Check help message for supported types..")

    args = parser.parse_args()

    print(args.file)
    
    if args.type == "model":
        model = models.load_model(args.file)

    elif args.type == "weight":
        weight = np.load("output/model_weights.npz")
        print(weight.files)
        
        print(weight['layer5'].shape)
        print(weight['layer6'].shape)
        print(weight['layer7'].shape)
        print(weight['layer8'].shape)
        print(weight['layer9'].shape)
        print(weight['layer10'].shape)

