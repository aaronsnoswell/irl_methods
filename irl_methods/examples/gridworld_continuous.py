# -*- coding: utf-8 -*-
"""
Continuous gridworld implementation from 'Algorithms for Inverse Reinforcement
Learning' by Ng and Russell, 2000

Copyright 2018 Aaron Snoswell
"""


import math
import numpy as np

import gym
from gym.utils import seeding


class GridWorldCtsEnv(gym.Env):
    """A continuous GridWorld MDP

    Based on the GridWorld described in 'Algorithms for Inverse Reinforcement
    Learning' by Ng and Russell, 2000
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }


    # Edge mode static enum
    EDGE_MODE_CLAMP = 0
    EDGE_MODE_WRAP = 1

    # Sytax sugar enum for actions
    ACTION_NORTH = 0
    ACTION_EAST = 1
    ACTION_SOUTH = 2
    ACTION_WEST = 3


    def __init__(
        self,
        *,
        action_distance = 0.2,
        wind_range = 0.1,
        edge_mode = EDGE_MODE_CLAMP,
        initial_state = (0.1, 0.1),
        goal_range = ((0.8, 0.8), (1, 1)),
        per_step_reward = 0,
        goal_reward = 1
        ):
        """
        Constructor for the GridWorld environment

        This MDP uses an x-first, y-up coordinate system
        """

        assert edge_mode == GridWorldCtsEnv.EDGE_MODE_WRAP \
            or edge_mode == GridWorldCtsEnv.EDGE_MODE_CLAMP, \
                "Invalid edge_mode: {}".format(edge_mode)
        
        # How far one step takes the agent
        self._action_distance = action_distance

        # Wind range (how far the wind can push the user each time step)
        self._wind_range = wind_range

        # Edge of world behaviour
        self._edge_mode = edge_mode

        # Set of actions
        self._A = [
            # North
            (0, 1),

            # East
            (1, 0),

            # South
            (0, -1),

            # West
            (-1, 0),

        ]

        # Gym visualisation object                        
        self.viewer = None

        # Gym action space object (index corresponds to entry in self._A list)
        self.action_space = gym.spaces.Discrete(len(self._A))

        # Gym observation space object
        self.observation_space = gym.spaces.Box(
            low=np.array((0, 0)),
            high=np.array((1, 1)),
            dtype=float
        )

        # Starting state
        self._initial_state = initial_state
        self.state = self._initial_state

        # Goal state
        self._goal_space = gym.spaces.Box(
            low=np.array(goal_range[0]),
            high=np.array(goal_range[1]),
            dtype=float
        )

        # Per step reward
        self._per_step_reward = per_step_reward

        # Goal reward
        self._goal_reward = goal_reward

        # Reset the MDP
        self.seed()
        self.reset()


    def seed(self, seed = None):
        """
        Set the random seed for the environment
        """

        self.np_random, seed = seeding.np_random(seed)

        return [seed]


    def step(self, action):
        """
        Take one step in the environment
        """

        assert self.action_space.contains(action), \
            "%r (%s) invalid" % (action, type(action))

        # Copy current state
        new_state = np.array(self.state, dtype=float)

        # Apply user action
        new_state += np.array(self._A[action]) * self._action_distance

        # Apply wind
        new_state += np.random.uniform(low=0, high=self._wind_range, size=2)

        # Apply boundary condition
        if self._edge_mode == GridWorldCtsEnv.EDGE_MODE_WRAP:
            self.state = new_state % 1.0

        else:
            self.state = tuple(map(lambda a: min(max(0, a), 1), new_state))

        # Check if we're done or not
        done = self._goal_space.contains(np.array(self.state))

        # Compute reward
        reward = self._per_step_reward
        if done:
            reward += self._goal_reward 

        # As per the Gym.Env definition, return a (s, r, done, status) tuple
        return self.state, reward, done, {}


    def reset(self):
        """
        Reset the environment to it's initial state
        """

        self.state = self._initial_state
        
        return self.state


    def render(self, mode = 'human'):
        """
        Render the environment

        TODO ajs 29/Apr/18 Implement viewer functionality
        """

        """
        screen_width = 600
        screen_height = 400

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
        
        return self.viewer.render(return_rgb_array = (mode == 'rgb_array'))
        """

        return None


    def close(self):
        """
        Close the environment
        """

        if self.viewer:
            self.viewer.close()

