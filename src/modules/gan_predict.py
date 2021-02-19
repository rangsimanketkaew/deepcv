"""
Deep learning-based collective variables (DeepCV)
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
from tensorflow.keras.models import load_model


def generate_latent_points(latent_dim, n_samples):    
    """Generate points in the latent spaces and reshape the vector

    Args:
        latent_dim (int): Dimension of latent space
        n_samples (int): Number of samples
    
    Returns:
        latent_space (array): Points in the latent spaces
    """
    latent_space = np.random.randn(latent_dim * n_samples)
    latent_space = latent_space.reshape(n_samples, latent_dim)
    return latent_space

if __name__ == "__main__":
    info = "Generate output using Generator's model."
    parser = argparse.ArgumentParser(description=info)
    parser.add_argument(
        "--model", "-m", metavar="MODEL", type=str, required=True, 
        help="Generator's model."
        )
    parser.add_argument(
        "--latent-dim", "-d", metavar="LATENT_DIM", type=int, required=True,
        help="Dimension of latent space."
        )
    parser.add_argument(
        "--num-samples", "-n", metavar="NUM_SAMPLESKEY", type=int, required=True,
        help="Number of samples."
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
    model = load_model(args.model)
    X = model.predict(vector)
