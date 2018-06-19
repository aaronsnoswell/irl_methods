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
        (list): A single trajectory, as list of (s, a, r) pairs
    """

    # Reset the MDP and set the initial state
    mdp.reset()
    mdp.state = start_state

    trajectory = []
    while True:

        # Copy starting state
        start_state = mdp.state

        # Query the policy for an action
        action = policy(mdp.state)

        # Take that action
        state, reward, done, status = mdp.step(action)

        # Store the (s, a, r) tuple
        trajectory.append((start_state, action, reward))

        # Check exit conditions
        if done or len(trajectory) > max_length:

            # Append the final state
            trajectory.append((mdp.state, None, None))

            break

    return trajectory
