# -*- coding: utf-8 -*-
"""A gridworld example for the Linear Programming IRL algorithm

Copyright 2018 Aaron Snoswell
"""

# Get the irl_methods folder on our PATH
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from lp import lp
from gridworld import GridWorldEnv

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def main():

    """
    This example re-creates the 5x5 gridworld experiment from the original
    Ng and Russell paper
    """

    # Construct a gridworld
    N = 5
    gw = GridWorldEnv(N=N)
    
    # The expert policy used in Ng and Russell, 2000
    expert_policy = [
        ['→', '→', '→', '→', ' '],
        ['↑', '→', '→', '↑', '↑'],
        ['↑', '↑', '↑', '↑', '↑'],
        ['↑', '↑', '→', '↑', '↑'],
        ['↑', '→', '→', '→', '↑'],
    ]

    # Get a sorted transition matrix
    T = gw.order_transition_matrix(expert_policy)

    gamma = 0.9

    # Plot various LP IRL results against each other

    fig = plt.figure()
    plt.set_cmap("viridis")

    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(1, 3),
        axes_pad=0.15,
        share_all=True,
        cbar_location="right",
        cbar_mode="single",
        cbar_size="7%",
        cbar_pad=0.15,
    )

    font_size = 15

    for ai, ax in enumerate(grid):

        plt.sca(ax)

        if ai == 0:
            gw.plot_reward(gw._R)
            plt.title("True Reward", fontsize=font_size)

        elif ai == 1:

            l1 = 0
            reward, _ = lp(T, gamma, l1=l1)
            gw.plot_reward(reward)
            plt.title(r"IRL Result - $\lambda$={}".format(l1), fontsize=font_size)

        elif ai == 2:

            l1 = 1.05
            reward, _ = lp(T, gamma, l1=l1)
            gw.plot_reward(reward)
            plt.title(r"IRL Result - $\lambda$={}".format(l1), fontsize=font_size)

    # Add colorbar
    plt.colorbar(cax=grid[0].cax)

    plt.show()


if __name__ == "__main__":

    main()
