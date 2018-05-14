# -*- coding: utf-8 -*-
"""A gridworld example for the Linear Programming for large state spaces IRL
algorithm

Copyright 2018 Aaron Snoswell
"""

# Get the irl_methods folder on our PATH
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import math
import numpy as np

from lpl import lpl
from basis import gaussian, indicator
from gridworld_continuous import GridWorldCtsEnv

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def main():
    """This example tests the linear programming for large state spaces IRL
    algorithm on a continuous gridworld problem
    """

    # Construct the continuous gridworld from the Ng and Russell paper
    N = 5
    cell_size = 1 / N
    goal_pos = (0.9, 0.9)

    gw = GridWorldCtsEnv(
        action_distance = 0.2,
        wind_range = 0.1,
        edge_mode = GridWorldCtsEnv.EDGE_MODE_CLAMP,
        initial_state = (cell_size/2, cell_size/2),
        goal_range = (
            (goal_pos[0] - cell_size/2, goal_pos[0] - cell_size/2),
            (goal_pos[0] + cell_size/2, goal_pos[0] + cell_size/2)
        ),
        per_step_reward = 0,
        goal_reward = 1
    )

    # Get expert policy for this gridworld
    expert_policy = gw.get_optimal_policy()

    k = len(gw._A)


    def T(s, ai):
        """Sampling transition function encoding the expert policy

        Args:
            s (tuple): Current state
            ai (int): Index of the action to be sampled. 0 must correspond to
                the expert policy action, and the other indices must be an
                arbitrary but consistent ordering of actions.

        Returns:
            (tuple): Next state
        """

        # Ask expert for an action
        expert_action = expert_policy(s)

        # Re-order the list of actions to have the expert's choice first
        A = list(range(len(gw._A)))
        del A[expert_action]
        A = [expert_action] + A

        # Now get the requested action
        action = A[ai]

        # Sample the MDP to get the next state
        gw.reset()
        gw.unwrapped.state = s
        state, reward, done, info = gw.step(action)
        return state


    s0 = [
        np.array((cell_size/2 + x/N, cell_size/2 + y/N))
            for y in range(N) for x in range(N)
    ]

    # Define a set of 2D circular gaussian basis functions
    sigma = 0.125
    #phi = [gausisan(s, np.diag(sigma, sigma)) for s in s0]
    phi = [indicator(s, np.array((cell_size, cell_size))) for s in s0]

    # Run IRL
    alpha_vector, _ = lpl(s0, k, T, phi, N=100, p=2.0, verbose=True)

    print(alpha_vector)

    # Compose reward function lambda
    R = lambda s: np.dot(alpha_vector, [phi[i](s) for i in range(len(phi))])[0]

    fig = plt.figure()
    plt.set_cmap("viridis")

    # Raster an image from the reward function for plotting
    img_size = 5
    img = np.empty(shape=(img_size, img_size), dtype=float)
    for y in range(img_size):
        for x in range(img_size):
            # Raster this pixel
            img[img_size - y - 1, x] = R(
                (
                    (1/img_size)/2 + x/img_size,
                    (1/img_size)/2 + y/img_size
                )
            )

    #np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    #print(img)

    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.show()


if __name__ == "__main__":

    main()
