# -*- coding: utf-8 -*-
"""Implementation of Deep Maximum-entropy IRL by Wulfmeier, Ondrùška and
Posner, 2016

Copyright 2018 Aaron Snoswell
"""

import math
import numpy as np


def deep_maximum_entropy(
    expert_state_action_frequencies,
    *,
    verbose = False
    ):
    """Deep Maximum-entropy IRL by Wulfmeier, Ondrùška and Posner, 2016
    
    Args:
        expert_state_action_frequencies (numpy array): Numpy 2D array of the
            expert's state-action frequency counts. First dimension
            corresponds to state index, second dimension to action index.

        verbose (bool): Show progress information

    Returns:

    """

    # Sum over actions to compute the expert's state counts
    expert_state_counts = np.sum(expert_state_action_frequencies, 2)


    # Initialize weight vector

    # Number of model refinement iterations
    N = 1000

    # Iterative model refinement
    for n in range(N):

        # Find current reward estimate
        r_n = nn_forward(f, theta_n)

        # Find MDP solution with current reward
        pi_n = approximate_value_iteration(r_n, S, A, T, gamma)
        feature_expectations = policy_propagation(pi_n, S, A, T)

        # Determine maximum entropy loss and gradients
        data_loss = log(pi_n) * mu_a_D
        gradient_of_data_loss_wrt_current_reward = mu_D - feature_expectations

        # Compute network gradients


    raise NotImplementedError


def approximate_value_iteration():
    raise NotImplementedError


def policy_propagation():
    raise NotImplementedError

