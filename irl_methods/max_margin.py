# -*- coding: utf-8 -*-
"""Implementation of max-margin IRL by Abbeel and Ng, 2004

Copyright 2018 Aaron Snoswell
"""

import math
import numpy as np

from thirdparty.robsvm import robsvm


def feature_expectations(trajectories, phi, gamma):
    """Compute the feature expectations for a collection of trajectories
    
    Args:
        trajectories (iterator): A generator/iterator/list that yields
            trajectories, each of which is a list of states
        phi (function): Function taking a state and returning a feature vector
            as a numpy array
        gamma (float): Discount factor

    Returns:
        (numpy array): A list of the averaged discounted feature expectations
            over the given trajectories

    """

    feature_expectations = np.zeros(shape=(phi(None)))

    # Loop over trajectories
    for trajectory in trajectory_iterator:

        # Loop over times and states in each trajectory
        for t, state in enumerate(trajectory):

            # Add up the discounted features at each step
            expert_feature_expectations += phi(state) * gamma ** t

    # Return the average expectations
    return feature_expectations / len(trajectory_iterator)


def sample_trajectories(start_states, policy, step, max_trajectory_length):
    """Generates a collection of trajectories given a policy

    Args:
        start_states (list): A list of seed states to use as the start of each
            trajectory. Length defines how many trajectories to sample.
        policy (function): Function mapping states to actions
        step (function): Function that evaluates an action and returns a new
            state
        max_trajectory_length (int): Maximum length of a trajectory

    Returns:
        (list): A list of sampled trajectories (each a list of states)
    """

    trajectories = []
    for trajectory_i, start_state in enumerate(start_states):

        trajectory = [start_state]
        t = 1
        while t < max_trajectory_length:
            state = trajectory[-1]
            action = policy(state)
            
            next_state = step(state, action)
            if next_state == None:
                break
            
            trajectory.append(next_state)
            t += 1

        trajectories.append(trajectory)

    return trajectories


def robust_svm_fit(X, y):
    """Fit an SVM hyperplane robustly

    After fitting, the hyperplane can be visualised (in 2D) by plotting the
    line y = mx * c where m = -w[0] / w[1] and c = -b / w[1].

    Args:
        X (numpy array): A 2D array of SVM sample points. First dimension is
            number of samples, second dimension is dimensionality of each
            sample. In our case, each sample is a feature vector.
        y (numy array): Vector of sample labels (in our case 1 is expert
            policy, -1 is non-expert policy)
    
    Returns:
        w (numpy array): Weight vector for the discovered hyperplane
        b (float): Bias for the hyperplane
    """

    m = x.shape[0]
    n = x.shape[1]

    X = matrix(X)
    labels = matrix(y)
    gamma = 1

    P = [matrix(np.eye(n))]
    e = matrix(np.array([0] * m))

    w, b, u, v, iterations = robsvm(X, labels, gamma, P, e)

    return w, b


def max_margin(
    trajectories,
    phi,
    gamma,
    initial_policy,
    step,
    rl_solver,
    *,
    start_states = None,
    max_trajectory_length = None,
    verbose = False
    ):
    """Maximum-margin IRL by Abbeel and Ng, 2004
    
    Args:
    
    Returns:
    """

    # Measure size of feature space
    k = phi(None)

    # Measure number of trajectories
    m = len(trajectories)

    # If start states are not specified, use those in the expert
    # demonstrations
    if start_states == None:
        start_states = [t[0] for t in trajectories]

    # If no max trajectory length is given, use the max length in the
    # demonstrations
    if max_trajectory_length == None:
        max_trajectory_length = max([len(t) for t in trajectories])

    # Compute the emperical feature expectations for the expert's trajectories
    expert_feature_expectations = feature_expectations(
        trajectories,
        phi,
        gamma
    )

    # A list of non-expert policies and their associated feature expectations
    nonexpert_policies = [initial_policy]
    nonexpert_feature_expectations = np.zeros(shape=(0, k))

    # Initialize the weight and reward function variables
    w = None
    reward_function = None

    # Loop until convergence
    while True:

        # Get latest policy
        policy = nonexpert_policies[-1]

        # Sample some new trajectories
        trajectories = sample_trajectories(
            start_states,
            policy,
            step,
            max_trajectory_length
        )

        # Compute feature expectations of the new policy and add to the list
        nonexpert_feature_expectations = np.vstack(
            (
                nonexpert_feature_expectations,
                feature_expectations(
                    trajectories,
                    phi,
                    gamma
                )
            )
        )

        # Solve the SVM / QP problem to find our current error
        X = np.vstack(
            (
                expert_feature_expectations,
                nonexpert_feature_expectations
            )
        )
        y = np.array([1] + [-1] * len(nonexpert_feature_expectations))
        w, b = robust_svm_fit(X, y)

        # Lambda for the current reward estimate
        reward_function = lambda s: np.matmul(w, phi(s))

        # Check the error of the current hyperplane, and break if we're close
        # to the expert's feature expectations
        t = np.linalg.norm(w)
        if t < epsillon:
            break

        # Compute a new optimal policy using the current reward function
        new_policy = rl_solver(reward_function)
        nonexpert_policies.append(new_policy)

        # Loop

    # We now have a list of policies, and by blending between the most recent
    # two, we can get a policy that achieves very similar feature expectations
    # to the expert policy
    return w, \
        reward_function, \
        nonexpert_policies, \
        nonexpert_feature_expectations

