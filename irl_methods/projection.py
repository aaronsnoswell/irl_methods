# -*- coding: utf-8 -*-
"""Implementation of Projection IRL by Abbeel and Ng, 2004

Copyright 2018 Aaron Snoswell
"""

import math
import numpy as np

from .thirdparty import robust_svm_fit


def find_mixing_weights(mu_e, mu_i):
    """Find mixing weights to match expert feature expectations

    Solves the quadratic programming problem of finding mixing weights
    'lambda' for mu_i such that the sum over lambda · mu_i is very close to
    mu_e (measured by L2-norm).
    
    Args:
        mu_e (numpy array): A feature expectation vector that we are trying
            to match (e.g. that of the expert).
        mu_i (numpy array): A numpy array of i vertically stacked sub-optimal
            feature expectation vectors
    
    Returns:
        (numpy array): A weight vector 'lambda' such that the sum of all
            lambda · mu_i is close to mu_e
    """

    raise NotImplementedError


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


def projection(
    trajectories,
    phi,
    gamma,
    initial_policy,
    step,
    rl_solver,
    epsillon,
    *,
    method = "projection",
    start_states = None,
    max_trajectory_length = None,
    verbose = False
    ):
    """Projection IRL by Abbeel and Ng, 2004
    
    Args:
        trajectories (list): A list of expert demonstration trajectories, each
            of which is a list of states (e.g. a numpy array)
        phi (function): A function that takes a state and returns a vector of
            features. NB: If the given state is None, must return the size
            of the feature space.
        gamma (float): Expert discount factor
        initial_policy (function): Initial policy to seed the algorithm. A
            function taking a state and returning an action.
        step (function): MDP step function taking a state and action, and
            returning a new state.
        rl_solver (function): An RL solver that takes a reward function (that
            takes a state and returns a reward), and returns an optimal
            policy function.
        epsillon (float): Convergence criteria - when the feature expectation
            error gets below this value, the algorithm will terminate.
        
        method (string): One of 'projection' or 'max-margin'. Determins the
            update method to use for finding feature expectation errors.
            'projection' will use the projection method outlined in the paper,
            whereas 'max-margin' will use the original SVM / Quadratic
            Programming update method (more computationally expensive).
        start_states (list): A list of seed states to use as the start of each
            trajectory. Length defines how many trajectories to sample when
            computing Mote-carlo estimates of the feature expectation for
            each non-expert policy. If not given, the expert demonstration
            starting states are used.
        max_trajectory_length (int): The maximum length to consider when
            computing feature expectations for the expert and non-expert
            trajectories. If not given, the length of the longest expert
            trajectory is used.
        verbose (bool): Show progress information

    Returns:
        (numpy array): Weights for the optimal reward function. Reward
            function can be computed as w · phi(s)
        (list): A list of the sub-optimal policy functions discovered during
            execution. Blending between the last two elements in the list will
            provide a policy with very close feature expectations to the
            expert demonstrations.
        (list): A list of the feature expectations for each sub-optimal policy
            discovered during execution.
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
        [t[0:max_trajectory_length] for t in trajectories],
        phi,
        gamma
    )

    # A list of non-expert policies and their associated feature expectations
    nonexpert_policies = [initial_policy]
    nonexpert_feature_expectations = np.zeros(shape=(0, k))
    nonexpert_blended_feature_expectations = np.zeros(shape=(0, k))

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

        if method == 'projection':

            # Solve the SVM / QP problem to find our current error
            X = np.vstack(
                (
                    expert_feature_expectations,
                    nonexpert_feature_expectations
                )
            )
            y = np.array([1] + [-1] * len(nonexpert_feature_expectations))
            w, b = robust_svm_fit(X, y)

        else:

            # Use the projection method to find our current error
            mu_e = expert_feature_expectations
            mu_prev = nonexpert_feature_expectations[-1]

            mu_prev_prev = nonexpert_feature_expectations[-2]
            mu_bar_prev_prev = nonexpert_blended_feature_expectations[-2, :]

            # The below finds the orthogonal projection of the expert's
            # feature expectations onto the line through mu_prev and
            # mu_prev_prev
            mu_bar_prev = mu_bar_prev_prev \
                + (mu_prev - mu_bar_prev_prev).T \
                    @ (mu_e - mu_bar_prev_prev) \
                / (mu_prev - mu_bar_prev_prev).T \
                    @ (mu_prev - mu_bar_prev_prev) \
                * (mu_prev - mu_bar_prev_prev)

            nonexpert_blended_feature_expectations = np.vstack(
                (
                    nonexpert_blended_feature_expectations,
                    mu_bar_prev
                )
            )

            w = mu_e - mu_bar_prev


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
        nonexpert_policies, \
        nonexpert_feature_expectations

