# -*- coding: utf-8 -*-
"""
Simple value iteration implementation

Copyright 2018 Aaron Snoswell
"""

import math
import numpy as np


def value_iteration(
        states,
        transition_tensor,
        policy,
        reward,
        discount,
        *,
        tol=1e-6,
        max_iterations=None
):
    """Find a value function given a policy and reward function

    Args:
        states (list): List of states to use when estimating the value function
        transition_tensor (numpy array) Transition matrix T[s, a, s']
            indicating the probability of arriving in state s' from state s,
            if you take action a.
        policy (function): Function pi(s) -> a_i mapping states to action
            integers
        reward (function): Function r(s) -> float mapping states to rewards
        discount (float): Discount factor to use when computing the value
            function

        tol (float): Stopping tolerance - when no state values change by more
            than this, the value refinement will cease
        max_iterations (int): Maximum iterations, or None to run until
            convergence

    Return:
        (numpy array): Vector of value estimates for each of the states provided
    """

    max_iterations = max_iterations if max_iterations is not None else math.inf

    values = np.zeros(len(states))
    reward_vector = np.array([reward(s) for s in states], dtype=float)

    # Loop until convergence or max iterations reached
    i = 0
    while True:
        i += 1

        delta = 0
        for si, state in enumerate(states):
            previous_value = values[si]

            # Get next action
            ai = policy(state)

            # Apply bellman equation
            p = transition_tensor[si, ai, :]
            values[si] = sum(p * (reward_vector + discount * values))

            # Update delta
            delta = max(delta, abs(values[si] - previous_value))

        # Check termination conditions
        if delta < tol or i > max_iterations:
            break

    return values


def demo():
    """Demonstrate functions from this module
    """

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from irl_methods.mdp import GridWorldDiscEnv

    # Construct a gridworld
    size = 5
    gw = GridWorldDiscEnv(
        size=size
    )
    states = list(range(size * size))
    transition_tensor = gw.transition_tensor
    policy = gw.get_optimal_policy()

    def reward(s):
        """Reward function r(s) -> float

        Args:
            s (any): State

        Returns:
            (float): Reward for the given state
        """
        assert gw.observation_space.contains(s), "Invalid state: {}".format(s)

        return gw.ground_truth_reward[s]

    # Discount factor
    discount = 0.9

    # Compute value estimate
    print("Doing value iteration...")
    values = value_iteration(
        states,
        transition_tensor,
        policy,
        reward,
        discount
    )
    print("Done, V =")
    print(np.flipud(np.reshape(values, (size, size))))

    fig = plt.figure()
    plt.set_cmap("viridis")
    plt.suptitle('Value Iteration demo')

    ax = plt.subplot(1, 3, 1)
    gw.plot_reward(ax, gw.ground_truth_reward, r_min=0, r_max=1)
    plt.title(r"Reward Function $R(s)$")
    plt.colorbar(
        cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    )

    ax = plt.subplot(1, 3, 2)
    gw.plot_policy(ax, policy)
    plt.title(r"Policy $\pi(s)$")

    ax = plt.subplot(1, 3, 3)
    gw.plot_reward(ax, values, r_min=0)
    plt.title(r"Value Function $V^\pi(s)$")
    plt.colorbar(
        cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    )

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo()
