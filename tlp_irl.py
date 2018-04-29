"""
Implementation of Trajectory-based Linear Programming IRL by Ng and Russell,
2000
(c) 2018 Aaron Snoswell
"""

import math
import warnings
import numpy as np

from cvxopt import matrix, solvers


def tlp_irl(zeta, T, S_bounds, A, phi, gamma, opt_pol, *, p=2.0, m=5000, H=30,
        tol=1e-6, verbose=False, on_iteration=None):
    """
    Implements trajectory-based Linear Programming IRL by Ng and Russell, 2000
    
    @param zeta - A list of lists of state, action tuples. The expert's
        demonstrated trajectories provided to the algorithm
    @param T - A sampling transition function T(s, a) -> s' that returns a new
        state after applying action a from state s (and also returns a bool
        indicating if the MDP has terminated)
    @param S_bounds - List of tuples indicating the bounds of the state space
    @param A - The MDP action space
    @param phi - A vector of d basis functions phi_i(s) mapping from S to real
        numbers
    @param gamma - Discount factor for future rewards
    @param opt_pol - A function f(alpha) -> (f(s) -> a) That returns an
        optimal policy function, given an alpha vector defining a linear
        reward function

    @param p - Penalty function coefficient. Ng and Russell find p=2 is robust
        Must be >= 1
    @param m - The number of trajectories to roll out when estimating
        emperical policy values
    @param H - The length at which to truncate trajectories
    @param tol - Float tolerance used to determine reward function convergence
    @param verbose - Print status information
    @param on_iteration - Optional callback function f(alpha) to be called
        after each LP iteration

    @return A vector of d 'alpha' coefficients for the basis functions phi(S)
        that allows rewards to be computed for a state via the inner product
        alpha_i · phi
    @return Final result object from the LP optimiser
    """

    # Measure length of basis function set
    d = len(phi)

    # Get starting state s0
    start_state = zeta[0][0][0]

    # Check for consistent starting states
    for demonstration in range(len(zeta) - 1):
        if demonstration[0] != start_state:
            warnings.warn("The starting state is not consistent across expert \
                demonstrations. TLP IRL may still work, but the algorithm \
                assumes a consistent starting state s0")
            break


    def state_as_index_tuple(s, N):
        """
        Discretises the given state into it's appropriate indices, as a tuple
        
        @param s - The state to discretise into indices
        @param N - The number of discretisation steps to use for each state
            dimension
        """
        indices = []
        for state_index, state in enumerate(s):
            s_min, s_max = S_bounds[state_index]
            state_disc_index = round((state - s_min) / (s_max - s_min) * (N-1))
            state_disc_index = min(max(state_disc_index, 0), N-1)
            indices.append(int(state_disc_index))
        return tuple(indices)


    def random_policy(N=20):
        """
        Generates a uniform random policy function by discretising the state
        space

        @param N - The number of discretisation steps to use for each state
            dimension
        """
        r_policy = np.random.choice(A, [N] * len(S_bounds))
        return (
            lambda policy:
                lambda s: policy[state_as_index_tuple(s, N)]
            )(r_policy)


    def mc_trajectories(policy, T, phi, *, m=m, H=H):
        """
        Sample some monte-carlo trajectories from the given policy tensor

        @param policy - A function f(s) -> a that returns the next action as a
            function of our current state
        @param T - A sampling transition function function T(s, a) -> s'
            that returns a new state after applying action a from state s
            (and also returns a bool indicating if the MDP has terminated)
        @param phi - A vector of d reward function basis functions mapping
            from S to real numbers
        @param m - The number of trajectories to roll out
        @param H - The length at which to truncate trajectories
        """
        trajectories = []
        for i in range(m):

            state = start_state
            rollout = []

            while len(rollout) < H:
                action = policy(state)
                rollout.append((state, action))

                state, done = T(state, action)
                if done: break
            trajectories.append(rollout)

        return trajectories


    def emperical_policy_value(zeta, phi=phi, gamma=gamma):
        """
        Computes the vector of mean discounted future basis function values
        for the given set of trajectories representing a single policy

        @param zeta - A list of lists of state, action tuples
        @param phi - Vector of basis functions defining a linear reward
            function
        @param gamma - Discount factor
        """


        def emperical_trajectory_value(trajectory, phi=phi, gamma=gamma):
            """
            Computes the vector of discounted future basis function values for
            the given trajectory. The inner product of this vector and an
            alpha vector gives the emperical value of the trajectory under the
            reward function defined by alpha

            @param zeta - A list of lists of state, action tuples
            @param phi - Vector of basis functions defining a linear reward
                function
            @param gamma - Discount factor
            """
            value = np.zeros(shape=d)
            for i in range(d):
                phi_i = phi[i]
                for j in range(len(trajectory)):
                    value[i] += gamma ** j * phi_i(trajectory[j][0])
            return value


        value_vector = np.zeros(shape=d)
        for trajectory in zeta:
            value_vector += emperical_trajectory_value(trajectory)
        value_vector /= len(zeta)

        return value_vector


    # Initialize alpha vector
    alpha = np.random.uniform(size=d) * 2 - 1

    # Compute mean expert trajectory value vector
    expert_value_vector = emperical_policy_value(zeta)

    # Initialize the non-expert policy set with a random policy
    non_expert_policy_set = [random_policy()]
    non_expert_policy_value_vectors = np.array([
        emperical_policy_value(
            mc_trajectories(
                non_expert_policy_set[-1],
                T,
                phi
            )
        )
    ])

    # Formulate the linear programming problem constraints
    # NB: The general form for adding a constraint looks like this
    # c, A_ub, b_ub = f(c, A_ub, b_ub)


    def add_optimal_expert_constraints(c, A_ub, b_ub):
        """
        Add constraints that enforce that the expert demonstrations are better
        than all other known policies (of which there are k)

        This will add k new optimisation variables and 2*k constraints

        NB: Assumes the true optimisation variables are first in the c vector
        """

        k = len(non_expert_policy_set)

        # Step 1: Add optimisation variables for each known non-expert policy
        c = np.hstack([np.zeros(shape=(1, d)), np.ones(shape=(1, k))])
        A_ub = np.hstack([A_ub, np.zeros(shape=(A_ub.shape[0], k))])

        # Step 2: Add constraints to ensure the expert demonstration is better
        # than each non-expert policy (nb: we use np ufunc broadcasting here)

        # Add first half of penalty function
        A_ub = np.vstack([
            A_ub,
            np.hstack([
                expert_value_vector - non_expert_policy_value_vectors,
                -1 * np.identity(k)
            ])
        ])
        b_ub = np.vstack([b_ub, np.zeros(shape=(k, 1))])

        # Add second half of penalty function
        A_ub = np.vstack([
            A_ub,
            np.hstack([
                p * (expert_value_vector - non_expert_policy_value_vectors),
                -1 * np.identity(k)
            ])
        ])
        b_ub = np.vstack([b_ub, np.zeros(shape=(k, 1))])

        
        return c, A_ub, b_ub


    def add_alpha_size_constraints(c, A_ub, b_ub):
        """
        Add constraints for a maximum |alpha| value of 1

        This will add 2 * d extra constraints

        NB: Assumes the true optimisation variables are first in the c vector
        """
        for i in range(d):
            constraint_row = [0] * A_ub.shape[1]
            constraint_row[i] = 1
            A_ub = np.vstack((A_ub, constraint_row))
            b_ub = np.vstack((b_ub, 1))

            constraint_row = [0] * A_ub.shape[1]
            constraint_row[i] = -1
            A_ub = np.vstack((A_ub, constraint_row))
            b_ub = np.vstack((b_ub, 1))
        return c, A_ub, b_ub


    # Variable to store LP optimisier result
    result = None

    # Loop until reward convergence
    while True:

        # Prepare LP constraint matrices
        if verbose: print("Composing LP problem...")
        c = np.zeros(shape=[1, d], dtype=float)
        A_ub = np.zeros(shape=[0, d], dtype=float)
        b_ub = np.zeros(shape=[0, 1])

        # Compose LP optimisation problem
        c, A_ub, b_ub = add_optimal_expert_constraints(c, A_ub, b_ub)
        c, A_ub, b_ub = add_alpha_size_constraints(c, A_ub, b_ub)

        # Solve the LP problem
        if verbose:
            print("Solving LP problem...")
            print("Number of optimisation variables: {}".format(c.shape[1]))
            print("Number of constraints: {}".format(A_ub.shape[0]))

        # NB: cvxopt.solvers.lp expects a 1d c vector
        solvers.options['show_progress'] = verbose
        res = solvers.lp(matrix(c[0, :]), matrix(A_ub), matrix(b_ub))

        # Extract the true optimisation variables
        alpha_new = res['x'][0:d].T

        if verbose: print("Done")

        if on_iteration is not None:
            on_iteration(alpha_new)

        # If alpha_i's have converged, break
        alpha_delta = np.linalg.norm(alpha - alpha_new)
        if verbose: print("Alpha delta: {}".format(alpha_delta))
        if alpha_delta <= tol:
            if verbose: print("Got reward convergence")
            break

        # Move alphas closer to true reward
        alpha = alpha_new

        # Find a new optimal policy based on the new alpha vector and add it
        # to the list of known policies
        if verbose: print("Finding new optimal policy")
        non_expert_policy_set.append(opt_pol(alpha))
        non_expert_policy_value_vectors = np.vstack([
            non_expert_policy_value_vectors,
            emperical_policy_value(
                mc_trajectories(
                    non_expert_policy_set[-1],
                    T,
                    phi
                )
            )
        ])


        # Loop


    return alpha, result



if __name__ == "__main__":

    # Collect a single trajectory from the human user
    print("Collecting expert trajectory")
    from gym_mountaincar import run_episode, manual_policy
    _, _, zeta = run_episode(manual_policy)

    # Create tmp MC object so we can read properties of the MDP
    import gym
    env = gym.make('MountainCarContinuous-v0')


    def T(state, action):
        """
        A sampling trasition function that allows us to simulate forwards
        through the world dynamics
        """
        env.reset()
        env.unwrapped.state = state
        state, reward, done, status = env.step([action])
        return state, done


    def normal(mu, sigma, x):
        """
        1D Normal function
        """
        return math.exp(-1 * ((x - mu) ** 2) / (2 * sigma ** 2)) / \
            math.sqrt(2 * math.pi * sigma ** 2)


    # Build a set of basis functions
    d = 5
    sigma = 0.4
    min_pos = env.unwrapped.min_position
    max_pos = env.unwrapped.max_position
    delta = (max_pos - min_pos) / d
    phi = [
        (lambda mu: lambda s: normal(mu, sigma, s[0]))(p) for p in np.arange(
            min_pos + delta/2,
            max_pos + delta/2,
            delta
        )
    ]

    # Solve the MDP using state, action discretisation for funtion
    # approximation
    print("Preparing discrete MDP approximation")
    from mountain_car import ContinuousMountainCar
    from mdp.policy import Policy, UniformRandomPolicy
    mc_mdp = ContinuousMountainCar(env, gamma=0.9, N=(20, 5))


    def opt_pol(alpha):
        """
        Finds an optimal policy given an alpha vector defining a linear
        reward function
        """

        # Find optimal policy via policy iteration
        v_star, p_star = Policy.policy_iteration(
            UniformRandomPolicy(mc_mdp),
            {s: 1/len(mc_mdp.state_set) for s in mc_mdp.state_set}
        )


        def nearest_in_list(x, lst):
            """
            Helper function to find the nearest entry in lst to x
            """
            nearest_index = -1
            nearest_dist = math.inf
            for li, i in enumerate(lst):
                dist = np.linalg.norm(np.array(x) - np.array(i))
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_index = li
            return lst[nearest_index]


        def pol(s):
            # First discretised the given state
            s_disc = nearest_in_list(s, list(p_star.policy_mapping.keys()))
            
            # Look up the action options for this state
            action_dict = p_star.policy_mapping[s_disc]

            # Sample an action and return it
            return np.random.choice(list(action_dict.keys()), p=list(action_dict.values()))

        return pol


    # Visualise reward function as we go
    import matplotlib.pyplot as plt
    

    def visualise(alpha_vector, block=False):

        print(alpha_vector)

        # Compose reward function
        R = lambda s: np.dot(alpha_vector, [phi[i](s) for i in range(len(phi))])[0]

        # Show basis functions and reward function
        x = np.linspace(env.unwrapped.min_position, env.unwrapped.max_position, 100)

        plt.clf()
        for i in range(len(alpha_vector)):
            plt.plot(
                x,
                list(map(lambda s: alpha_vector[i] * phi[i]([s, 0]), x)),
                'r--'
            )

        y = np.array(list(map(lambda s: R([s, 0]), x)))
        plt.plot(
            x,
            y,
            'b'
        )

        plt.grid()
        plt.title(r"Reward function $R(\pi^*)$")
        plt.xlim([env.unwrapped.min_position, env.unwrapped.max_position])
        if block:
            plt.show()
        else:
            plt.pause(0.05)


    # Perform trajectory based IRL
    # NB: MountainCar is not stocahstic, so we only need to sample m=1
    # trajectory to estimate trajectory value
    alpha_vector, res = tlp_irl(
        [zeta],
        T,
        [
            (env.unwrapped.min_position, env.unwrapped.max_position),
            (-env.unwrapped.max_speed, env.unwrapped.max_speed)
        ],
        [-1, 0, 1],
        phi,
        0.9,
        opt_pol,
        m=1,
        verbose=True,
        on_iteration=visualise
    )

    visualise(alpha_vector, block=True)

