# -*- coding: utf-8 -*-
"""Mountain Car example for the Linear Programming for large state spaces IRL
algorithm

Copyright 2018 Aaron Snoswell
"""

# Get the irl_methods folder on our PATH
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import copy
from lpl import lpl
import gym


"""
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
        # Helper function to find the nearest entry in lst to x
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
"""


def main():
    """This example re-creates the continuous mountain car experiment from the
    original Ng and Russell paper
    """

    # Construct an IRL problem from the MountainCar benchmark
    env = gym.make('MountainCar-v0')

    # Lambda that returns i.i.d. samples from state space
    sf = lambda: [
        np.random.uniform(env.unwrapped.min_position, \
            env.unwrapped.max_position),
        np.random.uniform(-env.unwrapped.max_speed, \
            env.unwrapped.max_speed)
    ]

    # Number of states to use for reward function estimation
    M = 5000

    # There are three possible actions
    # 0 = Full left force
    # 1 = Coast
    # 2 = Full right force
    A = [0, 1, 2]
    k = len(A)


    # Transition function
    def T(s, action_index, expert_policy):
        """
        Sampling transition function encoding the expert's policy

        That is, T[s, 0] takes the expert action and T[s, i] takes the ith
        non-expert action
        """

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
        observation, reward, done, info = env.step(action)
        return observation


    # A transition function for a simple 'expert' policy that solves the
    # mountain car problem by running 'bang-bang' control based on the
    # velocity (with a bias term)
    bb_policy = lambda s: 0 if (s[1] - 0.003) < 0 else 2
    T_bb = lambda s, ai: T(s, ai, bb_policy)


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
    min_pos = env.unwrapped.min_position
    max_pos = env.unwrapped.max_position
    delta = (max_pos - min_pos) / d
    sigma = delta * 0.25
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
    alpha_vector, res = lpl(sf, M, k, T_bb, phi, verbose=True)
    print(alpha_vector)

    # Compose reward function lambda
    R = lambda s: np.dot(alpha_vector, [phi[i](s) for i in range(len(phi))])[0]

    # Produce a nice plot to show the discovered reward function
    import matplotlib.pyplot as plt
    fig = plt.figure()
    x = np.linspace(env.unwrapped.min_position, env.unwrapped.max_position, 500)

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


if __name__ == "__main__":

    main()
