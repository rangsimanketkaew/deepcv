#!/usr/bin/env python3

"""
Deep Learning for Collective Variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
28/11/2020 : Rangsiman Ketkaew
"""

"""Optimize hyperparameter using Bayesian method
"""

from functools import partial
from bayes_opt import BayesianOptimization

fit_with_partial = partial(train, param_1, param_2)
# baye_optimizer will take care of other parameters of function that will be fine-tuned

# Example of bounded region of parameter space
pbounds = {"dropout2_rate": (0.1, 0.5), "lr": (1e-4, 1e-2)}

optimizer = BayesianOptimization(
    f=fit_with_partial,
    pbounds=pbounds,
    verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)

# fine tune parameter to maximize accuracy of model
optimizer.maximize(
    init_points=10,
    n_iter=10,
)


for i, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res))

print(optimizer.max)
