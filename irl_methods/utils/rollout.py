# -*- coding: utf-8 -*-
"""Roll out a policy to generate trajectories

Copyright 2018 Aaron Snoswell
"""

import math


def rollout(mdp, start_state, policy, *, max_length=math.inf):
    """Roll out a policy to generate (s, a) trajectories

    Args:
        mdp (gym.env): MDP instance
        start_state (any): Starting state for the rollout
        policy (function): Policy p(s) -> a

        max_length: Maximum trajectory length

    Returns:
        (list): A single trajectory, as list of (s, a) pairs
        (list): A list of the rewards received after each action
    """

    state = start_state
    trajectory = []
    rewards = []
    while True:
        # Query the policy for an action
        action = policy(state)

        # Store the (s, a) tuple
        trajectory.append((state, action))

        # Take that action
        mdp.reset()
        mdp.state = state
        state, reward, done, status = mdp.step(action)
        rewards.append(reward)

        # Check exit conditions
        if done or len(trajectory) > max_length:
            break

    # Add final action
    trajectory.append((state, None))

    return trajectory, rewards
