# -*- coding: utf-8 -*-
"""Implementation of Linear Programming IRL for large state spaces by Ng and
Russell, 2000

Copyright 2018 Aaron Snoswell
"""

import math
import numpy as np

from cvxopt import matrix, solvers


def lpl(sf, M, k, T, phi, *, N=1, p=2.0, verbose=False):
    """Linear Programming IRL for large state spaces by Ng and Russell, 2000

    Given a sampling transition function T(s, a_i) -> s' encoding a stationary
    deterministic policy and a set of basis functions phi(s) over the state
    space, finds a weight vector alpha for which the given policy is optimal
    with respect to R(s) = alpha · phi(s).

    See https://thinkingwires.com/posts/2018-02-13-irl-tutorial-1.html> for a
    good overview of this method.
    
    Args:
        sf (function): A 'state factory' function that takes no arguments and
            returns an i.i.d. sample from the MDP state space
        M (int): The number of sub-samples to draw from the state space when
            estimating the expert's reward function (|S_0|)
        k (int): The number of actions (|A|)
        T (function): A sampling transition function T(s, a_i) -> s' encoding
            a stationary deterministic policy. The structure of T must be that
            the 0th action T(:, 0) corresponds to a sample from the expert
            policy, and T(:, i), i!=0 corresponds to a sample from the ith
            non-expert action at each state, for some arbitrary but consistent
            ordering of actions.
        phi (list of functions): A vector of basis functions phi_i(s) that
            take a state and give a float.

        N (int): Number of transition samples to use when computing
            expectations. For deterministic MDPs, this can be left as 1.
        p (float): Penalty function coefficient. Ng and Russell find p=2 is
            robust. Must be >= 1.
        verbose (bool): Print progress information
    
    Returns:
        A tuple containing;
            - A numpy array of alpha coefficients for the basis functions.
            - A result object from the LP optimiser
    """

    # Measure number of basis functions
    d = len(phi)

    # Enforce valid penalty function coefficient
    assert p >= 1, \
        "Penalty function coefficient must be >= 1, was {}".format(p)


    def expectation(fn, sf, N):
        """
        Helper function to estimate an expectation over some function fn(sf())

        @param fn - A function of a single variable that the expectation will
            be computed over
        @param sf - A state factory function - takes no variables and returns
            an i.i.d. sample from the state space
        @param N - The number of draws to use when estimating the expectation

        @return An estimate of the expectation E[fn(sf())]
        """
        state = sf()
        return sum([fn(sf()) for n in range(N)]) / N


    # Measure number of basis functions
    d = len(phi)

    # Precompute the value expectation tensor VE
    # This is an array of shape (d, k-1, M) where VE[:, i, j] is a vector of
    # coefficients that, when multiplied with the alpha vector give the
    # expected difference in value between the expert policy action and the
    # ith action from state s_j
    VE_tensor = np.zeros(shape=(d, k-1, M))

    # Draw M initial states from the state space
    for j in range(M):
        if j % max(int(M/20), 1) == 0 and verbose:
            print("Computing expectations... ({:.1f}%)".format(j/M*100))

        s_j = sf()

        # Compute E[phi(s')] where s' is drawn from the expert policy
        expert_basis_expectations = np.array([
            expectation(phi[di], lambda: T(s_j, 0), N) for di in range(d)
        ])

        # Loop over k-1 non-expert actions
        for i in range(1, k):

            # Compute E[phi(s')] where s' is drawn from the ith non-expert action
            ith_non_expert_basis_expectations = np.array([
                expectation(phi[di], lambda: T(s_j, i), N) for di in range(d)
            ])

            # Compute and store the expectation difference for this initial
            # state
            VE_tensor[:, i-1, j] = expert_basis_expectations - \
                ith_non_expert_basis_expectations

    
    # TODO ajs 06/Jun/18 Remove redundant and trivial VE_tensor entries as
    # they create duplicate constraints

    # Formulate the linear programming problem constraints
    # NB: The general form for adding a constraint looks like this
    # c, A_ub, b_ub = f(c, A_ub, b_ub)
    if verbose: print("Composing LP problem...")


    def add_costly_single_step_constraints(c, A_ub, b_ub):
        """
        Augments the objective and adds constraints to implement the Linear
        Programming IRL method for large state spaces

        This will add up to M extra variables and 2M*(k-1) constraints
        (it does not add 'trivial' constraints)

        NB: Assumes the true optimisation variables are first in the c vector
        """

        # Step 1: Add the extra optimisation variables for each min{} operator
        # (one per sampled state)
        c = np.hstack([np.zeros(shape=(1, d)), np.ones(shape=(1, M))])
        A_ub = np.hstack([A_ub, np.zeros(shape=(A_ub.shape[0], M))])

        # Step 2: Add the constraints

        # Loop for each of the starting sampled states s_j
        for j in range(VE_tensor.shape[2]):
            if j % max(int(M/20), 1) == 0 and verbose:
                print("Adding constraints... ({:.1f}%)".format(j/M*100))

            # Loop over the k-1 non-expert actions
            for i in range(1, k):

                # Add two constraints, one for each half of the penalty
                # function p(x)
                constraint_row = np.hstack([VE_tensor[:, i-1, j], \
                    np.zeros(shape=M)])
                constraint_row[d + j] = -1
                A_ub = np.vstack((A_ub, constraint_row))
                b_ub = np.vstack((b_ub, 0))

                constraint_row = np.hstack([p * VE_tensor[:, i-1, j], \
                    np.zeros(shape=M)])
                constraint_row[d + j] = -1
                A_ub = np.vstack((A_ub, constraint_row))
                b_ub = np.vstack((b_ub, 0))

        return c, A_ub, b_ub


    def add_alpha_size_constraints(c, A_ub, b_ub):
        """
        Add constraints for a maximum |alpha| value of 1
        This will add 2 * d extra constraints

        NB: Assumes the true optimisation variables are first in the c vector
        """
        for i in range(d):
            constraint_row = [0] * A_ub.shape[1]
            constraint_row[i] = 1
            A_ub = np.vstack((A_ub, constraint_row))
            b_ub = np.vstack((b_ub, 1))

            constraint_row = [0] * A_ub.shape[1]
            constraint_row[i] = -1
            A_ub = np.vstack((A_ub, constraint_row))
            b_ub = np.vstack((b_ub, 1))
        return c, A_ub, b_ub


    # Prepare LP constraint matrices
    c = np.zeros(shape=[1, d], dtype=float)
    A_ub = np.zeros(shape=[0, d], dtype=float)
    b_ub = np.zeros(shape=[0, 1])

    # Compose LP optimisation problem
    c, A_ub, b_ub = add_costly_single_step_constraints(c, A_ub, b_ub)
    c, A_ub, b_ub = add_alpha_size_constraints(c, A_ub, b_ub)

    if verbose:
        print("Number of optimisation variables: {}".format(c.shape[1]))
        print("Number of constraints: {}".format(A_ub.shape[0]))

    # Solve for a solution
    if verbose: print("Solving LP problem...")
    
    # NB: cvxopt.solvers.lp expects a 1d c vector
    solvers.options['show_progress'] = verbose
    res = solvers.lp(matrix(c[0, :]), matrix(A_ub), matrix(b_ub))

    # Extract the true optimisation variables
    alpha_vector = res['x'][0:d].T

    return alpha_vector, res

