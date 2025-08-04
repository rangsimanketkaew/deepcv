"""
Deep Learning for Collective Variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
28/11/2020 : Rangsiman Ketkaew
"""

"""
This code will use a trained generator model to generate (fake) samples 
from a given set of latent points (random noise/number).
"""

import argparse
import numpy as np
import tensorflow as tf
import logging

logging = logging.getLogger("DeepCV")


def generate_latent_points(latent_dim, n_samples):
    """Generate points in the latent spaces and reshape the vector and convert to tensor

    Args:
        latent_dim (int): Dimension of latent space
        n_samples (int): Number of samples

    Returns:
        latent_space (tensor): Points in the latent spaces
    """
    latent_space = np.random.randn(latent_dim * n_samples)
    latent_space = latent_space.reshape(n_samples, latent_dim)
    return tf.convert_to_tensor(latent_space, dtype=tf.float32)


def main():
    info = "Generate output using Generator's model."
    parser = argparse.ArgumentParser(description=info)
    parser.add_argument(
        "-m",
        "--model",
        dest="model",
        metavar="MODEL",
        type=str,
        required=True,
        help="A pretrained generative model.",
    )
    parser.add_argument(
        "-d",
        "--latent-dim",
        dest="latent_dim",
        metavar="LATENT_DIM",
        type=int,
        required=True,
        help="Dimension of the latent layer (festure representation).",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        dest="num_samples",
        metavar="NUM_SAMPLES",
        type=int,
        required=True,
        help="Number of samples.",
    )

    args = parser.parse_args()

    ######################################
    # Generate latent spaces for Generator
    ######################################
    # 1. Generate noise for one sample
    # The array of random number should has the same shape as input
    # vector = np.random.randn(100)
    # vector = vector.reshape(1, 100)

    # 2. Generate latent space fot multiple samples
    vector = generate_latent_points(args.latent_dim, args.num_samples)

    ############################
    # Predict (generate results)
    ############################
    logging.info("Imported a model and make a prediction")
    imported = tf.saved_model.load(args.model)
    model = imported.signatures["serving_default"]
    out_tensor = model(vector)
    logging.info(f"Shape of output tensor: {out_tensor['reshape'].shape}")
    logging.info("=" * 30 + " DONE " + "=" * 30)


if __name__ == "__main__":
    main()
