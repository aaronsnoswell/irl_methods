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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Edge mode enum
EDGEMODE_CLAMP = 0
EDGEMODE_WRAP = 1
EDGEMODE_STRINGS = [
    "Clamped",
    "Wrapped"
]

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

# Feature map enum
FEATUREMAP_COORD = 0
FEATUREMAP_IDENTITY = 1
FEATUREMAP_OTHER_DISTANCE = 2
FEATUREMAP_GOAL_DISTANCE = 3
FEATUREMAP_STRINGS = [
    "Coordinate",
    "Identity",
    "Other Distance",
    "Goal Distance"
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
        edge_mode=EDGEMODE_CLAMP,
        initial_state=(0, 0),
        goal_states=((4, 4),),
        per_step_reward=0,
        goal_reward=1
    ):
        """
        Constructor for the GridWorld environment

        NB: All methods and internal representations that work with GridWorld
        indices use an (x, y) coordinate system where +x is to the right and
        +y is up.

        @param size - The size of the grid world
        @param wind - The chance of a uniform random action being taken each
            step
        @param edge_mode - Edge of world behaviour, one of EDGEMODE_CLAMP or
            EDGEMODE_WRAP
        @param initial_state - Starting state as an (x, y) tuple for the agent
        @param goal_states - List of tuples of (x, y) goal states
        @param per_step_reward - Reward given every step
        @param goal_reward - Reward upon reaching the goal
        """

        assert edge_mode == EDGEMODE_WRAP \
               or edge_mode == EDGEMODE_CLAMP, \
            "Invalid edge_mode: {}".format(edge_mode)
        
        # Size of the gridworld
        self._size = size

        # Wind percentage (chance of moving randomly each step)
        self._wind = wind

        # Edge of world behaviour
        self._edge_mode = edge_mode

        # Set of states
        self._S = [
            (x, y) for y in range(self._size) for x in range(self._size)
        ]

        # Lambda to apply boundary condition to an x, y state
        self._apply_edge_mode = lambda x, y: (
                min(max(x, 0), self._size - 1) if
                (self._edge_mode == EDGEMODE_CLAMP)
                else x % size,
                min(max(y, 0), self._size - 1) if
                (self._edge_mode == EDGEMODE_CLAMP)
                else y % size,
            )

        # Lambda to get the state index of an x, y pair
        self._xy2s = lambda x, y: \
            self._apply_edge_mode(x, y)[1] * size + \
            self._apply_edge_mode(x, y)[0]

        # Lambda to convert a state to an x, y pair
        self._s2xy = lambda s: (s % self._size, s // self._size)

        # Set of actions
        # NB: The y direction is inverted so that the numpy array layout, when
        # shown using print(), matches the 'up is up' expectation
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

        # Transition matrix
        self._T = np.zeros(shape=(len(self._S), len(self._A), len(self._S)))

        # Loop over initial states
        for si, state in enumerate(self._S):

            # Get the initial state details
            x = state[0]
            y = state[1]

            # Loop over actions
            for ai, action in enumerate(self._A):

                # Get action details
                dx = action[0]
                dy = action[1]

                # Update probability for desired action
                self._T[
                    si,
                    ai,
                    self._xy2s(x + dx, y + dy)
                ] += (1 - wind)

                # Update probability for stochastic alternatives
                for wind_action in self._A:

                    wind_dx = wind_action[0]
                    wind_dy = wind_action[1]

                    self._T[
                        si,
                        ai,
                        self._xy2s(x + wind_dx, y + wind_dy)
                    ] += wind / len(self._A)

        # Gym action space object (index corresponds to entry in self._A list)
        self.action_space = gym.spaces.Discrete(len(self._A))

        # Gym observation space object (observations are an index indicating
        # the current state)
        self.observation_space = gym.spaces.Discrete(len(self._S))

        # Starting state
        self._initial_state = self._xy2s(*initial_state)
        self.state = self._initial_state

        # Goal states
        self._goal_states = [self._xy2s(*s) for s in goal_states]

        # Per step reward
        self._per_step_reward = per_step_reward

        # Goal reward
        self._goal_reward = goal_reward

        # Store true reward
        self._R = np.array([
            self._per_step_reward + self._goal_reward
            if s in self._goal_states
            else self._per_step_reward
            for s in range(len(self._S))
        ])

        # Check if we're done or not
        self._done = lambda: self.state in self._goal_states

        # Members used for rendering
        self._fig = None
        self._ax = None
        self._goal_patches = None
        self._state_patch = None

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

        # Sample subsequent state from transition matrix
        self.state = np.random.choice(
            range(len(self._S)),
            p=self._T[self.state, action, :]
        )

        # Check if we're done or not
        done = self._done()

        # Compute reward
        reward = self._per_step_reward
        if done:
            reward += self._goal_reward 

        # As per the Gym.Env definition, return a (s, r, done, status) tuple
        return self.state, reward, done, {}

    def close(self):
        """Cleans up
        """
        if self._fig is not None:
            # Close our plot window
            plt.close(self._fig)

        self._fig = None
        self._ax = None
        self._goal_patches = None
        self._state_patch = None

    def render(self, mode="human"):
        """
        Render the environment
        """

        assert mode in self.metadata["render.modes"], "Invalid render mode"

        if mode == "ansi":
            # Render the gridworld text at the console
            # We draw goal states as '$', and the current state as '@'
            # We also draw boundaries around the world as '#' if the edge mode
            # is clamp, or '+' if the edge mode is wrap

            ret = ""

            edge_symbol = "#" if self._edge_mode == EDGEMODE_CLAMP else "+"

            # Add top boundary
            ret += edge_symbol * (self._size + 2) + "\n"

            # Iterate over grid
            for y in range(self._size-1, -1, -1):

                # Add left boundary
                ret += edge_symbol

                # Draw gridworld objects
                for x in range(self._size):
                    s = self._xy2s(x, y)
                    if s == self.state:
                        ret += "@"
                    elif s in self._goal_states:
                        ret += "$"
                    else:
                        ret += " "

                # Add right boundary
                ret += edge_symbol

                # Add newline
                ret += "\n"

            # Add bottom boundary
            ret += edge_symbol * (self._size + 2)

            return ret

        elif mode == "human":
            # Render using a GUI

            cell_size = 1/self._size
            s2coord = lambda s: np.array(self._s2xy(s)) / self._size + \
                                (cell_size/2, cell_size/2)

            if self._fig is None:
                self._fig = plt.figure()
                self._ax = self._fig.gca()

                # Render the goal patch(es)
                self._goal_patches = []
                for g in self._goal_states:
                    goal_patch = mpatches.Rectangle(
                        s2coord(g) - (cell_size/2, cell_size/2),
                        1,
                        1,
                        color="green",
                        ec=None
                    )
                    self._goal_patches.append(goal_patch)
                    self._ax.add_patch(goal_patch)

                # Render the current position
                self._state_patch = mpatches.Circle(
                    s2coord(self.state),
                    0.025,
                    color="blue",
                    ec=None
                )
                self._ax.add_patch(self._state_patch)

                # Render a grid
                line_width = 0.75
                line_color = "#dddddd"
                for i in [x / self._size for x in range(self._size)]:
                    self._ax.add_artist(plt.Line2D(
                        (0, 1),
                        (i, i),
                        color=line_color,
                        linewidth=line_width
                    ))
                    self._ax.add_artist(plt.Line2D(
                        (i, i),
                        (0, 1),
                        color=line_color,
                        linewidth=line_width
                    ))

                self._ax.set_title(
                    "Discrete {} GridWorld, wind = {}".format(
                        EDGEMODE_STRINGS[self._edge_mode],
                        self._wind
                    )
                )
                self._ax.set_aspect(1)

                self._ax.set_xlim([0, 1])
                self._ax.set_xticks(
                    [s / self._size + cell_size/2 for s in range(self._size)]
                )
                self._ax.set_xticklabels(
                   [str(s + 1) for s in range(self._size)]
                )

                self._ax.set_ylim([0, 1])
                self._ax.set_yticks(
                    [s / self._size + cell_size/2 for s in range(self._size)]
                )
                self._ax.set_yticklabels(
                   [str(s + 1) for s in range(self._size)]
                )
                self._ax.xaxis.set_tick_params(size=0)
                self._ax.yaxis.set_tick_params(size=0)

            else:
                # We assume a stationary goal
                self._state_patch.center = np.array(s2coord(self.state))
                self._fig.canvas.draw()

            # Show the rendered GridWorld
            if self._done():
                # Plot and block
                plt.show()
            else:
                # Plot without blocking
                plt.pause(0.25)

            return None

        else:

            # Let super handle it
            super(GridWorldDiscEnv, self).render(mode=mode)

    def get_state_features(self, *, s=None, feature_map=FEATUREMAP_COORD):
        """Returns a feature vector for the given state

        Args:
            s (int): State integer, or None to use the current state
            feature_map (int): Feature map to use. One of FEATURE_MAP_*
                defined at the top of this module

        Returns:
            (numpy array): Feature vector for the given state
        """

        s = s or self.state

        if feature_map == FEATUREMAP_COORD:
            # Features are (x, y) coordinate tuples
            return np.array(self._s2xy(s))

        elif feature_map == FEATUREMAP_IDENTITY:
            # Features are zero arrays with a 1 indicating the location of
            # the state
            f = np.zeros(len(self._S))
            f[s] = 1
            return f

        elif feature_map == FEATUREMAP_OTHER_DISTANCE:
            # Features are an array indicating the L0 / manhattan distance to
            # each other state
            f = np.zeros(len(self._S))
            x0, y0 = self._s2xy(s)
            for other_state in range(len(self._S)):
                x, y = self._s2xy(other_state)
                f[other_state] = abs(x0 - x) + abs(y0 - y)
            return f

        elif feature_map == FEATUREMAP_GOAL_DISTANCE:
            # Features are an array indicating the L0 / manhattan distance to
            # each goal
            f = np.zeros(len(self._goal_states))
            x0, y0 = self._s2xy(s)
            for i, goal_state in enumerate(self._goal_states):
                x, y = self._s2xy(goal_state)
                f[i] = abs(x0 - x) + abs(y0 - y)
            return f

        else:
            assert False, "Invalid feature map: {}".format(feature_map)

    def order_transition_matrix(self, policy):
        """Computes a sorted transition matrix for the GridWorld MDP

        Given a policy, defined as either a function pi(s) -> a taking state
        integers and returning action integers, or as a 2D numpy array of
        unicode string arrows, computes a sorted transition matrix T[s, a,
        s'] such that the 0th action corresponds to the policy's action,
        and the ith action (i!=0) corresponds to the ith non-policy action,
        for some arbitrary but consistent ordering of actions.

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
            policy (numpy array) - Expert policy 'a1' as a function pi(s) ->
            a or a 2D numpy array. See the example above.

        Returns:
            A sorted transition matrix T[s, a, s'], where the 0th action
            T[:, 0, :] corresponds to following the expert policy, and the
            other action entries correspond to the remaining action options,
            sorted according to the ordering in GridWorldEnv._A

        """

        transitions_sorted = copy.copy(self._T)
        for y in range(self._size):
            for x in range(self._size):

                si = self._xy2s(x, y)
                action = None
                if callable(policy):
                    action = policy(si)
                else:
                    action = policy[y][x]
                    if action == '↑':
                        action = ACTION_NORTH
                    elif action == '→':
                        action = ACTION_EAST
                    elif action == '↓':
                        action = ACTION_SOUTH
                    elif action == '←':
                        action = ACTION_WEST

                if action == ACTION_NORTH:
                    # Expert chooses north
                    # North is already first in the GridWorldEnv._A ordering
                    pass
                elif action == ACTION_EAST:
                    # Expert chooses east
                    tmp = transitions_sorted[si, 0, :]
                    transitions_sorted[si, 0, :] = transitions_sorted[si, 1, :]
                    transitions_sorted[si, 1, :] = tmp
                elif action == ACTION_SOUTH:
                    # Expert chooses south
                    tmp = transitions_sorted[si, 0:1, :]
                    transitions_sorted[si, 0, :] = transitions_sorted[si, 2, :]
                    transitions_sorted[si, 1:2, :] = tmp
                elif action == ACTION_WEST:
                    # Expert chooses west
                    tmp = transitions_sorted[si, 0:2, :]
                    transitions_sorted[si, 0, :] = transitions_sorted[si, 3, :]
                    transitions_sorted[si, 1:3, :] = tmp
                else:
                    # Expert doesn't care / does nothing
                    pass

        return transitions_sorted

    def get_optimal_policy(self):
        """Returns an optimal policy function for this MDP

        Returns:
            (function): An optimal policy p(s) -> a that maps states to
                actions
        """

        # Collect goal states as (x, y) indices
        _goal_states = [self._s2xy(g) for g in self._goal_states]

        # If we're in a wrapping gridworld, the nearest goal could outside the
        # world bounds. Add virtual goals to help the policy account for this
        if self._edge_mode == EDGEMODE_WRAP:
            for i in range(len(self._goal_states)):
                g = np.array(self._s2xy(self._goal_states[i]))
                _goal_states.append(tuple(g - (self._size, self._size)))
                _goal_states.append(tuple(g - (0, self._size)))
                _goal_states.append(tuple(g - (self._size, 0)))

        def policy(state):
            """A simple expert policy to solve the continuous gridworld
            problem

            Args:
                state (tuple): The current MDP state integer

            Returns:
                (int): One of the ACTION_* globals defined in this module
            """

            # Pick the nearest goal
            nearest_goal = None

            if len(_goal_states) == 1:
                # We only have one goal to consider
                nearest_goal = _goal_states[0]

            else:
                # Find the nearest goal - it could be behind us
                smallest_distance = math.inf
                for g in _goal_states:
                    distance = np.linalg.norm(np.array(g) - self._s2xy(state))
                    if distance < smallest_distance:
                        smallest_distance = distance
                        nearest_goal = g

            # Find the distance to the goal
            dx, dy = np.array(nearest_goal) - self._s2xy(state)

            direction = None
            if abs(dx) == abs(dy):
                # x and y distances are equal - flip a coin to avoid bias in
                # the policy
                if np.random.uniform() > 0.5:
                    # Move vertically
                    direction = "vertically"
                else:
                    # Move horizontally
                    direction = "horizontally"

            if abs(dx) > abs(dy):
                # We need to move horizontally more than vertically
                direction = "horizontally"
            else:
                # We need to move vertically more than horizontally
                direction = "vertically"

            # Compute the actual movement
            if direction == "horizontally":
                return ACTION_EAST if dx > 0 \
                    else ACTION_WEST
            else:
                return ACTION_NORTH if dy > 0 \
                    else ACTION_SOUTH

        return policy

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
            edge_mode=EDGEMODE_CLAMP,
            initial_state=(0.1, 0.1),
            goal_range=((0.8, 0.8), (1, 1)),
            per_step_reward=0,
            goal_reward=1
    ):
        """Constructor for the GridWorld environment

        NB: All methods and internal representations that work with GridWorld
        coordinates use an (x, y) coordinate system where +x is to the right and
        +y is up.
        """

        assert edge_mode == EDGEMODE_WRAP \
               or edge_mode == EDGEMODE_CLAMP, \
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

        # Check if we're done or not
        self._done = lambda: self._goal_space.contains(np.array(self.state))

        # Members used for rendering
        self._fig = None
        self._ax = None
        self._goal_patch = None
        self._state_patch = None

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
        new_state += np.random.uniform(
            low=-self._wind_range,
            high=self._wind_range,
            size=2
        )

        # Apply boundary condition
        if self._edge_mode == EDGEMODE_WRAP:
            self.state = tuple(new_state % 1.0)

        else:
            self.state = tuple(map(lambda a: min(max(0, a), 1), new_state))

        # Check done-ness
        done = self._done()

        # Compute reward
        reward = self._per_step_reward
        if done:
            reward += self._goal_reward

        # As per the Gym.Env definition, return a (s, r, done, status) tuple
        return self.state, reward, done, {}

    def close(self):
        """Cleans up
        """
        if self._fig is not None:
            # Close our plot window
            plt.close(self._fig)

        self._fig = None
        self._ax = None
        self._goal_patch = None
        self._state_patch = None

    def render(self, mode='human'):
        """Render the environment
        """

        if mode == "human":
            # Render using a GUI

            if self._fig is None:
                self._fig = plt.figure()
                self._ax = self._fig.gca()

                # Render the goal patch
                self._goal_patch = mpatches.Rectangle(
                    self._goal_space.low,
                    self._goal_space.high[0] - self._goal_space.low[0],
                    self._goal_space.high[1] - self._goal_space.low[1],
                    color="green",
                    ec=None
                )
                self._ax.add_patch(self._goal_patch)

                # Render the current position
                self._state_patch = mpatches.Circle(
                    self.state,
                    0.025,
                    color="blue",
                    ec=None
                )
                self._ax.add_patch(self._state_patch)

                self._ax.set_title(
                    "Continuous {} GridWorld, wind={}".format(
                        EDGEMODE_STRINGS[self._edge_mode],
                        self._wind_range
                    )
                )
                self._ax.set_aspect(1)
                self._ax.set_xlim([0, 1])
                self._ax.set_ylim([0, 1])

            else:
                # We assume a stationary goal
                self._state_patch.center = self.state
                self._fig.canvas.draw()

            # Show the rendered GridWorld
            if self._done():
                # Plot and block
                plt.show()
            else:
                # Plot without blocking
                # plt.show(
                #     block=False
                # )
                plt.pause(0.25)

            return None

        else:
            # Let super handle it
            super(GridWorldCtsEnv, self).render(mode=mode)

    def get_state_features(self, *, s=None, feature_map=FEATUREMAP_COORD):
        """Returns a feature vector for the given state

        Args:
            s (numpy array): State as a numpy array, or None to use the
                current state
            feature_map (string): Feature map to use. One of
                FEATUREMAP_COORD or FEATUREMAP_GOAL_DISTANCE

        Returns:
            (numpy array): Feature vector for the given state
        """

        s = s or self.state

        if feature_map == FEATUREMAP_COORD:
            # Features are (x, y) coordinate tuples
            return np.array(s)

        elif feature_map == FEATUREMAP_GOAL_DISTANCE:
            # Feature is a value indicating the distance to the goal
            return np.linalg.norm(
                self._goal_space.low + self._goal_space.high - s
            )

        else:
            assert False, "Invalid feature map: {}".format(feature_map)

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
        if self._edge_mode == EDGEMODE_WRAP:
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
                (int): One of the ACTION_* globals defined in this module
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


if __name__ == "__main__":
    # Simple example of how to use these classes

    # Exercise discrete gridworld
    print("Testing discrete GridWorld...")
    gw_disc = GridWorldDiscEnv(per_step_reward=-1)
    policy = gw_disc.get_optimal_policy()

    print("Ordered transition matrix:")
    t_ordered = gw_disc.order_transition_matrix(policy)
    print(t_ordered)

    # Choose a feature map to use
    feature_map = FEATUREMAP_COORD
    print("Using feature map {}".format(FEATUREMAP_STRINGS[feature_map]))

    print(gw_disc.render(mode="ansi"))
    gw_disc.render(mode="human")
    reward = 0
    while True:
        action = policy(gw_disc.state)
        print("Observed f={}, taking action a={}".format(
            gw_disc.get_state_features(feature_map=feature_map),
            ACTION_STRINGS[action]
        ))
        s, r, done, status = gw_disc.step(action)
        print("Got reward {}".format(r))
        reward += r
        print(gw_disc.render(mode="ansi"))
        gw_disc.render(mode="human")

        if done:
            break

    gw_disc.close()
    print("Done, total reward = {}".format(reward))

    # Exercise cts gridworld
    print("Testing continuous GridWorld...")
    gw_cts = GridWorldCtsEnv(per_step_reward=-1)
    policy = gw_cts.get_optimal_policy()

    # Choose a feature map to use
    feature_map = FEATUREMAP_COORD
    print("Using feature map {}".format(FEATUREMAP_STRINGS[feature_map]))

    gw_cts.render()
    reward = 0
    while True:
        action = policy(gw_cts.state)
        print("Observed f={}, taking action a={}".format(
            gw_cts.get_state_features(feature_map=feature_map),
            ACTION_STRINGS[action]
        ))
        s, r, done, status = gw_cts.step(action)
        print("Got reward {}".format(r))
        reward += r
        gw_cts.render()

        if done:
            break

    gw_cts.close()
    print("Done, total reward = {}".format(reward))

