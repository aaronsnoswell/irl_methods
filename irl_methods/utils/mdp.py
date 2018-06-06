"""
Utilities for transforming MDP problems

(c) Aaron Snoswell 2018
"""


import numpy as np
import gym


def make_gridworld(S, trajectories, levels, *, dims=None):
    """Transforms continuous states and trajectories to a discrete gridworld

    Args:
        S (gym.spaces.Box): Continuous state space
        trajectories (list): List lists of (s, a) state action tuples
        levels (int): Number of discretisation levels per state dimension

        dims (list): Indices of the state dimensions to use in the
            discretisation, or None to use all dimensions

    Returns:
        (int): The resultant number of states
        (int): The resultant number of actions
        (gym.core.Env): Gridworld MDP environment
        (list): List of list of discretised (s, a) trajectories
        (function): Lambda to convert cts state to discrete
        (function): Lambda to convert discrete state to cts
        (function): Lambda providing an enumeration over gridworld states
        (function): Lambda converting an enumeration index to a gridworld
            state
        (function): Lambda providing an enumeration over gridworld actions
        (function): Lambda converting an enumeration index to a gridworld
            action
    """

    # Select only the relevant dimensions
    if dims == None:
        dims = list(range(len(S.low)))

    low = S.low[dims]
    high = S.high[dims]

    # Broadcast levels arg if necessary
    levels = levels * np.ones(low.shape)

    # Compute number of states
    n = int(np.prod(levels))

    # Copmute number of actions
    k = int(2 * len(levels))

    # Compute level deltas
    deltas = (high - low) / levels

    # Discretise a state
    s2disc = lambda s: np.array(
        np.maximum(
            np.minimum(
                (s[dims] - low) / deltas,
                levels-1
            ),
            0
        ),
        dtype=int
    )

    # Make a discrete state continuous
    s2cts = lambda s: np.array((s[dims] + 0.5) * deltas + low, dtype=float)


    # Fix an enumeration over gridworld states and actions
    def s2index(s):
        """Convert gridworld state to index
        """
        # Size of each subsequent dimension in state indices
        sz = [np.prod(list(levels[i:]) + [1]) for i in range(1, len(levels) + 1)]

        # Index is now the dot product of the sizes with the gridworld state
        # vector
        return int(s @ sz)


    def index2s(idx):
        """Convert index to gridworld state
        """
        # Size of each subsequent dimension in state indices
        sz = [np.prod(list(levels[i:]) + [1]) for i in range(1, len(levels) + 1)]

        s = np.zeros(len(levels), dtype=int)
        for dim in range(len(sz)):
            s[dim] = idx // sz[dim]
            idx -= s[dim] * sz[dim]
        return s


    def a2index(s):
        """Convert gridworld action to index

        [0 0] is 0
        [0 1] is 1
        [1 0] is 2
        [1 1] is 3
        [2 0] is 4
        ...
        """
        return s[0] * 2 + s[1]


    def index2a(idx):
        """Convert index to gridworld action

        0 is [0 0] (which is [-1  0  0 ...] as a delta vector)
        1 is [0 1] (which is [ 1  0  0 ...] as a delta vector)
        2 is [1 0] (which is [ 0 -1  0 ...] as a delta vector)
        3 is [1 1] (which is [ 0  1  0 ...] as a delta vector)
        4 is [2 0] (which is [ 0 -1  0 ...] as a delta vector)
        ...
        """
        return np.array([idx // 2, idx % 2 != 0], dtype=int)
        


    # Convert the trajectories
    trajectories_disc = []
    for traj in trajectories:

        # Discretise state trajectories
        state_traj_disc = []
        for s, a in traj:
            state_traj_disc.append(s2disc(s))

        # In-fill missing intermediate states
        state_traj_disc_infilled = []
        for start, end in zip(state_traj_disc[0:-1], state_traj_disc[1:]):
            # Compute the delta
            delta = end - start

            # Until we reach the current goal state
            current = start
            while np.any(abs(delta) > 0):

                # Add our current sate
                state_traj_disc_infilled.append(current)

                # Select a random dimension where we aren't yet aligned with
                # the goal (randomise to avoid bias)
                dim = -1
                while True:
                    dim = np.random.randint(0, len(current))
                    if delta[dim] != 0: break

                # Take a step in the selected direction
                step = np.zeros(shape=len(current), dtype=int)
                step[dim] = -1 if delta[dim] < 0 else 1
                current = current + step

                # Update the remaining delta
                delta -= step

        # Add the final goal state
        state_traj_disc_infilled.append(end)

        # Finally, compute the actions as deltas between states
        traj_disc = []
        for start, end in zip(
                state_traj_disc_infilled[0:-1],
                state_traj_disc_infilled[1:]
            ):
            action_delta_vector = end - start
            dim = np.nonzero(action_delta_vector)[0][0]
            sgn = 0 if action_delta_vector[dim] < 0 else 1
            action = [dim, sgn]
            traj_disc.append((start, action))

        traj_disc.append((end, None))

        trajectories_disc.append(traj_disc)

    # Construct a gym environment object
    # Env methods below

    def __init__(self):
        self.seed()
        self.reset()


    def seed(self):
        pass


    def reset(self):
        self._state = self.observation_space.sample()


    def _reward(self, state):
        # To be recovered by some IRL method
        return 0


    def _done(self):
        return False


    def step(self, action):
        if self._done(): return None

        # Actions are a 2-discrete vector indicating which dimension to move
        # in, and which direction (0 is 'negative', 1 is 'positive')
        # We convert this to a gridworld index delta vector
        action_delta_vector = np.zeros(
            self.action_space.nvec[0],
            dtype=int
        )
        action_delta_vector[action[0]] = -1 if action[1] == 0 else 1
        self._state += action_delta_vector

        return self._state, self._reward(self._state), self._done(), {}


    mdp = type(
        "DermatologyEMPGridworldEnv",
        (gym.core.Env, ),
        {
            "metadata": {'render.modes': []},
            "reward_range": (-np.inf, np.inf),
            "spec": None,
            "action_space": gym.spaces.MultiDiscrete([len(levels), 2]),
            "observation_space":  gym.spaces.MultiDiscrete(levels),
            "__init__": __init__,
            "seed": seed,
            "reset": reset,
            "_reward": _reward,
            "_done": _done,
            "step": step
        }
    )()

    return n, k, mdp, trajectories_disc, \
        s2disc, s2cts, s2index, index2s, a2index, index2a
