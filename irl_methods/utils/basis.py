# -*- coding: utf-8 -*-
"""Various basis function implementations

Copyright 2018 Aaron Snoswell
"""

import math
import numpy as np

from functools import partial


def gaussian(mu, sigma):
    """N-D gaussian implementation

    Args:
        mu (numpy array): Mean of the gaussian in N-D space
        sigma (numpy array): N x N Covariance matrix

    Returns:
        (float): A function f(x) that returns the value of the gaussian
            evaluated at x
    """

    # Measure size of space
    k = len(mu)

    # Straight-forward implementation for the 1-D case
    if k == 1:
        return lambda x: \
            math.exp(-1 * (x - mu) ** 2 / (2 * sigma ** 2)) / \
            math.sqrt(2 * math.pi * sigma ** 2)

    # Pre-compute inverse
    simga_inv = np.linalg.inv(sigma)

    # Pre-compute normalizer
    normalizer = 1.0 / math.sqrt(np.linalg.det(sigma) * (2 * math.pi) ** k)

    # Internal gaussian implementation
    def _gaussian(mu, sigma_inv, normalizer, x):
        return math.exp(-0.5 * np.matmul(np.matmul((x - mu), simga_inv),
                                         (x - mu))) * normalizer

    return partial(_gaussian, mu, simga_inv, normalizer)


def indicator(position, size):
    """N-D indicator function implementation

    Args:
        position (numpy array): Center of this indicator function in N-D space
        size (numpy array): Size of this indicator function's box in N-D space

    Returns:
        (function): A function f(x) that returns 1 if x is within an
            axis-aligned N-D box located at pos, and size units in every
            dimension, or returns 0 otherwise.
    """

    # Internal indicator function implementation
    def _indicator(position, size, x):
        return float(np.all(abs(position - x) < size / 2))

    return partial(_indicator, position, size)

