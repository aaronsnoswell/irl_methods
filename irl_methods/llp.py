# -*- coding: utf-8 -*-
"""Implementation of Linear Programming IRL for large state spaces by Ng and
Russell, 2000

Copyright 2018 Aaron Snoswell
"""

import math
import numpy as np

from cvxopt import matrix, solvers


def llp(sf, M, k, T, phi, *, N=1, p=2.0, verbose=False):
    """Linear Programming IRL for large state spaces by Ng and Russell, 2000

    Given a sampling transition function :math:`T(s, a_i) \\mapsto s'`
    encoding a stationary deterministic policy and a set of basis functions
    :math:`\\phi(s)` over the state space, finds a weight vector
    :math:`\\alpha` for which the given policy is optimal with respect to
    :math:`R(s) = \\alpha \\cdot \\phi(s)`.

    See `this blog post <https://thinkingwires.com/posts/2018-02-13-irl-tutorial-1.html>`_
    for a good overview.
    
    Args:
        sf (function): A 'state factory' function that takes no arguments and
            returns an i.i.d. sample from the MDP state space
        M (int): The number of sub-samples to draw from the state space when
            estimating the expert's reward function (:math:`|S_0|`)
        k (int): The number of actions (:math:`|A|`)
        T (function): A sampling transition function
            :math:`T(s, a_i) \\mapsto s'` encoding a stationary deterministic
            policy. The structure of T must be that the 0th action
            :math:`T(:, 0)` corresponds to a sample from the expert policy,
            and :math:`T(:, i), i \\ne 0` corresponds to a sample from the ith
            non-expert action at each state, for some arbitrary but consistent
            ordering of actions.
        phi (list of functions): A vector of basis functions
            :math:`\\phi_i(s)` that take a state and give a float.
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



def solve_mc():

    # Solve the MDP using state, action discretisation for funtion
    # approximation
    print("Preparing discrete MDP approximation")
    from mountain_car import ContinuousMountainCar
    mc_mdp = ContinuousMountainCar(env, gamma=0.99, N=(20, 5))

    # Compute optimal policy via PI
    print("Solving for optimal policy")
    from mdp.policy import Policy, UniformRandomPolicy
    v_star, p_star = Policy.policy_iteration(
        UniformRandomPolicy(mc_mdp),
        {s: 1/len(mc_mdp.state_set) for s in mc_mdp.state_set},
        verbose=True
    )

    # Try running the discovered optimal policy on the task
    from gym_mountaincar import run_episode


    def nearest_in_list(x, lst):
        """
        Helper function to find the nearest entry in lst to x
        """
        nearest_index = -1
        nearest_dist = math.inf
        for li, i in enumerate(lst):
            dist = np.linalg.norm(np.array(x) - np.array(i))
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_index = li
        return lst[nearest_index]


    def p_fn(observation, env, key_handler):
        # Discretise observation
        s_disc = nearest_in_list(observation, mc_mdp.state_set)
        return p_star.get_action(s_disc)


    run_episode(p_fn, continuous=True)



if __name__ == "__main__":

    """
    This example re-creates the continuous mountain car experiment from the
    original Ng and Russell paper
    """

    import copy
    import gym

    # Construct an IRL problem from the MountainCar benchmark
    env = gym.make('MountainCarContinuous-v0')

    # Lambda that returns i.i.d. samples from state space
    sf = lambda: [
        np.random.uniform(env.unwrapped.min_position, \
            env.unwrapped.max_position),
        np.random.uniform(-env.unwrapped.max_speed, \
            env.unwrapped.max_speed)
    ]

    # Number of states to use for reward function estimation
    M = 5000

    # Discretise the action space so there are three possible actions
    # 0 = Full left force
    # 1 = Coast
    # 2 = Full right force
    A = [0, 1, 2]
    k = len(A)


    # Transition function
    def T(s, action_index):
        """
        Sampling transition function encoding the expert's policy

        That is, T[s, 0] takes the expert action and T[s, i] takes the ith
        non-expert action
        """

        # A simple 'expert' policy that solves the mountain car problem by
        # running 'bang-bang' control based on the velocity (with a bias term)
        expert_policy = lambda s: 0 if (s[1] - 0.003) < 0 else 2

        # Sort the action set based on the expert's current action
        expert_action = expert_policy(s)
        A_non_expert = copy.copy(A)
        A_non_expert.remove(expert_action)
        A_sorted = [expert_action] + A_non_expert

        # Find the requested action
        action = A_sorted[action_index]

        # Reset and sample the environment to get the next state
        env.reset()
        env.unwrapped.state = s
        observation, reward, done, info = env.step([action])
        return observation


    # We use gaussian basis functions
    def normal(mu, sigma, x):
        """
        1D Normal function
        """

        return math.exp(-1 * ((x - mu) ** 2) / (2 * sigma ** 2)) / \
            math.sqrt(2 * math.pi * sigma ** 2)


    # Build basis function set of evenly spaced gaussians
    # Sigma was guestimated as it isn't reported in the original paper
    d = 26
    sigma = 0.2
    min_pos = env.unwrapped.min_position
    max_pos = env.unwrapped.max_position
    delta = (max_pos - min_pos) / d
    phi = [
        (lambda mu:
            lambda s:
                normal(mu, sigma, s[0])
        )(p) for p in np.arange(
            min_pos + delta/2,
            max_pos + delta/2,
            delta
        )
    ]

    
    # Run IRL
    alpha_vector, res = llp(sf, M, k, T, phi, verbose=True)
    print(alpha_vector)

    # Compose reward function lambda
    R = lambda s: np.dot(alpha_vector, [phi[i](s) for i in range(len(phi))])[0]

    # Produce a nice plot to show the discovered reward function
    import matplotlib.pyplot as plt
    fig = plt.figure()
    x = np.linspace(env.unwrapped.min_position, env.unwrapped.max_position, 100)

    # Plot basis functions
    for i in range(len(alpha_vector)):
        plt.plot(
            x,
            list(map(lambda s: alpha_vector[i] * phi[i]([s, 0]), x)),
            'r--'
        )

    # Plot reward function
    plt.plot(
        x,
        np.array(list(map(lambda s: R([s, 0]), x))),
        'b'
    )

    plt.grid()
    plt.title(r"Reward function $R(\pi^*)$")
    plt.xlim([env.unwrapped.min_position, env.unwrapped.max_position])
    plt.show()
