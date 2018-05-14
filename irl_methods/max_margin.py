# -*- coding: utf-8 -*-
"""Implementation of max-margin IRL by Abbeel and Ng, 2004

Copyright 2018 Aaron Snoswell
"""

import math
import numpy as np

from cvxopt import matrix, solvers


def max_margin(trajectories, gamma, k, phi, rl_solver, *, verbose=False):
    """Maximum-margin IRL by Abbeel and Ng, 2004
    
    Args:
    
    Returns:
    """

    # Measure number of trajectories
    m = len(trajectories)

    # Compute the emperical feature expectations for the expert's trajectories
    expert_feature_expectations = np.zeros(shape=(k))
    for i in range(m):
        trajectory = trajectories[i]
        for t in range(len(trajectory)):
            state_t = trajectory[t]
            expert_feature_expectations += phi(state_t) * gamma ** t
    expert_feature_expectations /= m


    # A list of the non-expert feature expectations
    nonexpert_feature_expectations = np.zeros(shape=(0, k))


    def add_optimal_expert_contraints(G, h):
        """Adds QP constraints to ensure the expert is optimal

        Assumes the 't' error term is first in the objective function,
        followed by the weight vector terms

        Args:
            G (numpy array): QP Vectorial inequality LHS constraint matrix
            h (numpy array): QP Vectorial inequality RHS vector

        Returns
            (numpy arrays): G and h, updated with new constraints
        """

        # Loop over our current set of less-than-expert policies
        for j in range(len(nonexpert_feature_expectations)):

            # For each policy, add one constraint that ensures the expert's
            # reward is greater than this policy's reward by at least a margin
            # of 't'
            G = np.vstack(
                (
                    G,
                    np.hstack(
                        (
                            1,
                            nonexpert_feature_expectations[j] \
                                - expert_feature_expectations
                        )
                    )
                )
            )
            h = np.vstack((h, 0))

        return G, h


