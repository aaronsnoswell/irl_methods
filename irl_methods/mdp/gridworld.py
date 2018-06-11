# -*- coding: utf-8 -*-
"""
Simple 2D GridWorld MDP implementation

Based on the GridWorld from 'Algorithms for Inverse Reinforcement Learning'
by Ng and Russell, 2000

Copyright 2018 Aaron Snoswell
"""

import math
import copy
import numpy as np
import gym
from gym.utils import seeding


# Edge mode static enum
EDGE_MODE_CLAMP = 0
EDGE_MODE_WRAP = 1

# Syntax sugar helpers for actions
ACTION_NORTH = 0
ACTION_EAST = 1
ACTION_SOUTH = 2
ACTION_WEST = 3
ACTION_STRINGS = [
    "North",
    "East",
    "South",
    "West"
]


class GridWorldDiscEnv(gym.Env):
    """A simple GridWorld MDP

    Based on the GridWorld described in 'Algorithms for Inverse Reinforcement
    Learning' by Ng and Russell, 2000
    """

    metadata = {
        'render.modes': ['human', 'ansi']
    }

    def __init__(
        self,
        *,
        size=5,
        wind=0.3,
        edge_mode=EDGE_MODE_CLAMP,
        initial_state=(4, 0),
        goal_states=((0, 4),),
        per_step_reward=0,
        goal_reward=1
    ):
        """
        Constructor for the GridWorld environment

        NB: All methods and internal representations use the y-first, y-down
        coordinate system

        @param size - The size of the grid world
        @param wind - The chance of a uniform random action being taken each
            step
        @param edge_mode - Edge of world behaviour, one of
            GridWorldEnv.EDGE_MODE_CLAMP or GridWorldEnv.EDGE_MODE_WRAP
        @param initial_state - Starting state for the agent
        @param goal_states - List of tuples of goal states
        @param per_step_reward - Reward given every step
        @param goal_reward - Reward upon reaching the goal
        """

        assert edge_mode == EDGE_MODE_WRAP \
            or edge_mode == EDGE_MODE_CLAMP, \
            "Invalid edge_mode: {}".format(edge_mode)
        
        # Size of the gridworld
        self._size = size

        # Wind percentage (chance of moving randomly each step)
        self._wind = wind

        # Edge of world behaviour
        self._edge_mode = edge_mode

        # Set of states
        # NB: We store y first, so that the numpy array layout, when shown
        # using print(), matches the 'y-is-vertical' expectation
        self._S = [
            (y, x) for y in range(self._size) for x in range(self._size)
        ]

        # Lambda to apply boundary condition to an x, y state
        self._apply_edge_mode = lambda x, y: (
                min(max(x, 0), self._size - 1) if
                (self._edge_mode == EDGE_MODE_CLAMP)
                else x % size,
                min(max(y, 0), self._size - 1) if
                (self._edge_mode == EDGE_MODE_CLAMP)
                else y % size,
            )

        # Lambda to get the state index of an x, y pair
        self._state_index = lambda x, y: \
            self._apply_edge_mode(x, y)[1] * size + \
            self._apply_edge_mode(x, y)[0]

        # Set of actions
        # NB: The y direction is inverted so that the numpy array layout, when
        # shown using print(), matches the 'up is up' expectation
        self._A = [
            # North
            (-1, 0),

            # East
            (0, 1),

            # South
            (1, 0),

            # West
            (0, -1),

        ]

        # Transition matrix
        self._T = np.zeros(shape=(len(self._S), len(self._A), len(self._S)))

        # Loop over initial states
        for si in [
                self._state_index(x, y)
                for x in range(self._size) for y in range(self._size)
        ]:

            # Get the initial state details
            state = self._S[si]
            x = state[1]
            y = state[0]

            # Loop over actions
            for ai in range(len(self._A)):

                # Get action details
                action = self._A[ai]
                dx = action[1]
                dy = action[0]

                # Update probability for desired action
                self._T[
                    si,
                    ai,
                    self._state_index(x + dx, y + dy)
                ] += (1 - wind)

                # Update probability for stochastic alternatives
                for wind_action in self._A:

                    wind_dx = wind_action[1]
                    wind_dy = wind_action[0]

                    self._T[
                        si,
                        ai,
                        self._state_index(x + wind_dx, y + wind_dy)
                    ] += wind / len(self._A)

        # Gym visualisation object                        
        self.viewer = None

        # Gym action space object (index corresponds to entry in self._A list)
        self.action_space = gym.spaces.Discrete(len(self._A))

        # Gym observation space object (observations are an index indicating
        # the current state)
        self.observation_space = gym.spaces.Discrete(self._size * self._size)

        # Starting state
        self._initial_state = initial_state
        self.state = self._initial_state

        # Goal states
        self._goal_states = goal_states

        # Per step reward
        self._per_step_reward = per_step_reward

        # Goal reward
        self._goal_reward = goal_reward

        # Store true reward
        self._R = np.array([
            self._per_step_reward + self._goal_reward
            if (y, x) in self._goal_states
            else self._per_step_reward
            for y in range(self._size) for x in range(self._size)
        ])

        # Reset the MDP
        self.np_random = None
        self.seed()
        self.reset()

    def seed(self, seed=None):
        """
        Set the random seed for the environment
        """

        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """
        Reset the environment to it's initial state
        """

        self.state = self._initial_state

        return self.state

    def step(self, action):
        """
        Take one step in the environment
        """

        assert self.action_space.contains(action), \
            "%r (%s) invalid" % (action, type(action))

        # Get current x and y coordinates
        x = self.state[1]
        y = self.state[0]

        # Sample subsequent state from transition matrix
        self.state = np.random.choice(
            range(self._size * self._size),
            p=self._T[self._state_index(x, y), action, :]
        )

        # Check if we're done or not
        done = self.state in self._goal_states

        # Compute reward
        reward = self._per_step_reward
        if done:
            reward += self._goal_reward 

        # As per the Gym.Env definition, return a (s, r, done, status) tuple
        return self.state, reward, done, {}

    def render(self, mode='human'):
        """
        Render the environment

        TODO ajs 29/Apr/18 Implement viewer functionality
        """

        return None

    def order_transition_matrix(self, policy):
        """Computes a sorted transition matrix for the GridWorld MDP

        Given a policy, defined as a 2D numpy array of unicode string arrows,
        computes a sorted transition matrix T[s, a, s'] such that the 0th
        action corresponds to the policy's action, and the ith action (i!=0)
        corresponds to the ith non-policy action, for some arbitrary but
        consistent ordering of actions.

        E.g.

        pi_star = [
            ['→', '→', '→', '→', ' '],
            ['↑', '→', '→', '↑', '↑'],
            ['↑', '↑', '↑', '↑', '↑'],
            ['↑', '↑', '→', '↑', '↑'],
            ['↑', '→', '→', '→', '↑'],
        ]

        is the policy used in Ng and Russell's 2000 IRL paper. NB: a space
        indicates a terminal state.

        Args:
            policy (numpy array) - Expert policy 'a1'. See the example above.

        Returns:
            A sorted transition matrix T[s, a, s'], where the 0th action
            T[:, 0, :] corresponds to following the expert policy, and the
            other action entries correspond to the remaining action options,
            sorted according to the ordering in GridWorldEnv._A

        """

        transitions_sorted = copy.copy(self._T)
        for y in range(self._size):
            for x in range(self._size):

                si = self._state_index(x, y)
                a = policy[y][x]

                if a == '↑':
                    # Expert chooses north
                    # North is already first in the GridWorldEnv._A ordering
                    pass
                elif a == '→':
                    # Expert chooses east
                    tmp = transitions_sorted[si, 0, :]
                    transitions_sorted[si, 0, :] = transitions_sorted[si, 1, :]
                    transitions_sorted[si, 1, :] = tmp
                elif a == '↓':
                    # Expert chooses south
                    tmp = transitions_sorted[si, 0:1, :]
                    transitions_sorted[si, 0, :] = transitions_sorted[si, 2, :]
                    transitions_sorted[si, 1:2, :] = tmp
                elif a == '←':
                    # Expert chooses west
                    tmp = transitions_sorted[si, 0:2, :]
                    transitions_sorted[si, 0, :] = transitions_sorted[si, 3, :]
                    transitions_sorted[si, 1:3, :] = tmp
                else:
                    # Expert doesn't care / does nothing
                    pass

        return transitions_sorted

    def plot_reward(self, rewards):
        """
        Plots a given reward vector
        """

        import matplotlib.pyplot as plt

        fig = plt.gcf()
        ax = plt.gca()

        line_color = "#efefef"

        plt.pcolor(
            np.reshape(rewards, (self._size, self._size)),
            edgecolors=line_color
        )
        plt.gca().invert_yaxis()
        
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(length=0, labelbottom=False, labelleft=False)

        # Figure is now ready for display or saving
        return fig


class GridWorldCtsEnv(gym.Env):
    """A continuous GridWorld MDP

    Based on the GridWorld described in 'Algorithms for Inverse Reinforcement
    Learning' by Ng and Russell, 2000
    """

    metadata = {
        'render.modes': ['human']
    }

    def __init__(
            self,
            *,
            action_distance=0.2,
            wind_range=0.1,
            edge_mode=EDGE_MODE_CLAMP,
            initial_state=(0.1, 0.1),
            goal_range=((0.8, 0.8), (1, 1)),
            per_step_reward=0,
            goal_reward=1
    ):
        """Constructor for the GridWorld environment

        This MDP uses an x-first, y-up coordinate system
        """

        assert edge_mode == EDGE_MODE_WRAP \
            or edge_mode == EDGE_MODE_CLAMP, \
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
        self.np_random = None
        self.seed()
        self.reset()

    def seed(self, seed=None):
        """
        Set the random seed for the environment
        """

        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """
        Reset the environment to it's initial state
        """

        self.state = self._initial_state

        return self.state

    def step(self, action):
        """Take one step in the environment
        """

        assert self.action_space.contains(action), \
            "%r (%s) invalid" % (action, type(action))

        # Copy current state
        new_state = np.array(self.state, dtype=float)

        # Apply user action
        new_state += np.array(self._A[action]) * self._action_distance

        # Apply wind
        new_state += np.random.uniform(low=-self._wind_range,
                                       high=self._wind_range, size=2)

        # Apply boundary condition
        if self._edge_mode == GridWorldCtsEnv.EDGE_MODE_WRAP:
            self.state = tuple(new_state % 1.0)

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

    def render(self, mode='human'):
        """Render the environment

        TODO ajs 29/Apr/18 Implement viewer functionality
        """

        return None

    def get_optimal_policy(self):
        """Returns an optimal policy function for this MDP

        Returns:
            (function): An optimal policy p(s) -> a that maps states to
                actions
        """

        # Compute goal location
        goal_positions = [np.mean(
            np.vstack(
                (
                    self._goal_space.low,
                    self._goal_space.high
                )
            ),
            axis=0
        )]

        # If we're in a wrapping gridworld, the nearest goal could outside the
        # world bounds. Add virtual goals to help the policy account for this
        if self._edge_mode == GridWorldCtsEnv.EDGE_MODE_WRAP:
            goal_positions.append(
                goal_positions[0] - (1, 1)
            )
            goal_positions.append(
                goal_positions[0] - (1, 0)
            )
            goal_positions.append(
                goal_positions[0] - (0, 1)
            )

        def policy(state):
            """A simple expert policy to solve the continuous gridworld
            problem

            Args:
                state (tuple): The current MDP state as an (x, y) tuple of
                    float

            Returns:
                (int): One of GridWorldCtsEnv.[ACTION_NORTH],
                    GridWorldCtsEnv.ACTION_EAST, GridWorldCtsEnv.ACTION_SOUTH,
                    or GridWorldCtsEnv.ACTION_WEST
            """

            # Pick the nearest goal
            nearest_goal = None

            if len(goal_positions) == 1:
                # We only have one goal to consider
                nearest_goal = goal_positions[0]

            else:
                # Find the nearest goal - it could be behind us
                smallest_distance = math.inf
                for g in goal_positions:
                    distance = np.linalg.norm(np.array(g) - state)
                    if distance < smallest_distance:
                        smallest_distance = distance
                        nearest_goal = g

            # Find the distance to the goal
            dx, dy = np.array(nearest_goal) - state

            if abs(dx) > abs(dy):
                # We need to move horizontally more than vertically
                return ACTION_EAST if dx > 0 \
                    else ACTION_WEST

            else:
                # We need to move vertically more than horizontally
                return ACTION_NORTH if dy > 0 \
                    else ACTION_SOUTH

        return policy
