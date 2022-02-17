#!/usr/bin/env python3

"""
Deep Learning for Collective Variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
11/06/2021 : Rangsiman Ketkaew
"""


def split_dataset(dataset, split_ratio=0.8):
    """Split dataset into traing and test sets

    Args:
        dataset (array): Dataset
        split_ratio (float): Split ratio (must be between 0.00 to 1.00). Default to 0.8

    Returns:
        [type]: [description]
    """
    # Determine size of actual training set
    size = int(split_ratio * dataset.shape[0])
    # Split dataset
    train, test = dataset[:size], dataset[size:]
    return train, test
