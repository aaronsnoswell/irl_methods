# -*- coding: utf-8 -*-
"""Implementation of Linear Programming IRL methods by Ng and Russell, 2000

Copyright 2018 Aaron Snoswell
"""

import warnings
import numpy as np
from cvxopt import matrix, solvers


def linear_programming(T, gamma, *, l1=0, Rmax=1.0, verbose=False):
    """Linear Programming IRL by NG and Russell, 2000

    Given a transition matrix T[s, a, s'] encoding a stationary
    deterministic policy and a discount factor gamma, finds a reward vector
    R(s) for which the policy is optimal.

    This method implements the `Linear Programming IRL algorithm by Ng and
    Russell <http://ai.stanford.edu/~ang/papers/icml00-irl.pdf>. See 
    <https://www.inf.ed.ac.uk/teaching/courses/rl/slides17/8_IRL.pdf> for an
    accessible overview.

    TODO: Adjust L1 norm constraint generation to allow negative rewards in
    the final vector.

    Args:
        T (numpy array): A sorted transition matrix T[s, a, s']
            encoding a stationary deterministic policy. The structure must be
            such that the 0th action T[:, 0, :] corresponds to the expert
            policy, and T[:, i, :], i != 0 corresponds to the ith non-expert
            action at each state.
        gamma (float): The expert's discount factor. Must be in the range
            [0, 1).
        
        l1 (float): L1 norm regularization weight for the LP optimisation
            objective function
        Rmax (float): Maximum reward value
        verbose (bool): Print progress information
    
    Returns:
        (numpy array): The reward vector for which the given policy is
            optimal, and
        (dict): A result object from the LP optimiser
    """

    # Measure size of state and action sets
    n = T.shape[0]
    k = T.shape[1]

    # Compute the discounted transition matrix inverse term
    T_disc_inv = np.linalg.inv(np.identity(n) - gamma * T[:, 0, :])

    # Formulate the linear programming problem constraints
    # NB: The general form for adding a constraint looks like this
    # c, A_ub, b_ub = f(c, A_ub, b_ub)
    if verbose:
        print("Composing LP problem...")

    # Prepare LP constraint matrices
    c = np.zeros(shape=[1, n], dtype=float)
    A_ub = np.zeros(shape=[0, n], dtype=float)
    b_ub = np.zeros(shape=[0, 1])

    def add_optimal_policy_constraints(c, A_ub, b_ub):
        """
        Add constraints to ensure the expert policy is optimal
        This will add (k-1) * n extra constraints
        """
        for i in range(k - 1):
            constraint_rows = -1 * (T[:, 0, :] - T[:, i, :]) @ T_disc_inv
            A_ub = np.vstack((A_ub, constraint_rows))
            b_ub = np.vstack((b_ub, np.zeros(shape=[constraint_rows.shape[0], 1])))
        return c, A_ub, b_ub

    def add_costly_single_step_constraints(c, A_ub, b_ub):
        """
        Augment the optimisation objective to add the costly-single-step
        degeneracy heuristic
        This will add n extra optimisation variables and (k-1) * n extra
        constraints
        NB: Assumes the true optimisation variables are first in the objective
        function
        """

        # Expand the c vector add new terms for the min{} operator
        c = np.hstack((c, -1 * np.ones(shape=[1, n])))
        css_offset = c.shape[1] - n
        A_ub = np.hstack((A_ub, np.zeros(shape=[A_ub.shape[0], n])))

        # Add min{} operator constraints
        for i in range(k - 1):
            # Generate the costly single step constraint terms
            constraint_rows = -1 * (T[:, 0, :] - T[:, i, :]) @ T_disc_inv

            # constraint_rows is nxn - we need to add the min{} terms though
            min_operator_entries = np.identity(n)
            
            # And we have to make sure we put the min{} operator entries in
            # the correct place in the A_ub matrix
            num_padding_cols = css_offset - n
            padding_entries = np.zeros(shape=[constraint_rows.shape[0], num_padding_cols])
            constraint_rows = np.hstack((constraint_rows, padding_entries, min_operator_entries))

            # Finally, add the new constraints
            A_ub = np.vstack((A_ub, constraint_rows))
            b_ub = np.vstack((b_ub, np.zeros(shape=[constraint_rows.shape[0], 1])))
        
        return c, A_ub, b_ub

    def add_l1norm_constraints(c, A_ub, b_ub, l1):
        """
        Augment the optimisation objective to add an l1 norm regularisation
        term z += l1 * ||R||_1
        This will add n extra optimisation variables and 2n extra constraints
        NB: Assumes the true optimisation variables are first in the objective
        function
        """

        # We add an extra variable for each each true optimisation variable
        c = np.hstack((c, l1 * np.ones(shape=[1, n])))
        l1_offset = c.shape[1] - n

        # Don't forget to resize the A_ub matrix to match
        A_ub = np.hstack((A_ub, np.zeros(shape=[A_ub.shape[0], n])))

        # Now we add 2 new constraints for each true optimisation variable to
        # enforce the absolute value terms in the l1 norm
        for i in range(n):

            # An absolute value |x1| can be enforced via constraints
            # -x1 <= 0             (i.e., x1 must be positive or 0)
            #  x1 + -xe1 <= 0
            # Where xe1 is the replacement for |x1| in the objective
            #
            # TODO ajs 04/Apr/2018 This enforces that R must be positive or 0,
            # but I was under the impression that it was also possible to
            # enforce an abs operator without this requirement - e.g. see
            # http://lpsolve.sourceforge.net/5.1/absolute.htm
            constraint_row_1 = [0] * A_ub.shape[1]
            constraint_row_1[i] = -1
            A_ub = np.vstack((A_ub, constraint_row_1))
            b_ub = np.vstack((b_ub, [[0]]))

            constraint_row_2 = [0] * A_ub.shape[1]
            constraint_row_2[i] = 1
            constraint_row_2[l1_offset + i] = -1
            A_ub = np.vstack((A_ub, constraint_row_2))
            b_ub = np.vstack((b_ub, [[0]]))

        return c, A_ub, b_ub

    def add_rmax_constraints(c, A_ub, b_ub, Rmax):
        """
        Add constraints for a maximum R value Rmax
        This will add n extra constraints
        """
        for i in range(n):
            constraint_row = [0] * A_ub.shape[1]
            constraint_row[i] = 1
            A_ub = np.vstack((A_ub, constraint_row))
            b_ub = np.vstack((b_ub, Rmax))
        return c, A_ub, b_ub
    
    # Compose LP optimisation problem
    c, A_ub, b_ub = add_optimal_policy_constraints(c, A_ub, b_ub)
    c, A_ub, b_ub = add_costly_single_step_constraints(c, A_ub, b_ub)
    c, A_ub, b_ub = add_rmax_constraints(c, A_ub, b_ub, Rmax)
    c, A_ub, b_ub = add_l1norm_constraints(c, A_ub, b_ub, l1)

    if verbose:
       print("Number of optimisation variables: {}".format(c.shape[1]))
       print("Number of constraints: {}".format(A_ub.shape[0]))

    # Solve for a solution
    if verbose:
        print("Solving LP problem...")

    # NB: cvxopt.solvers.lp expects a 1d c vector
    from cvxopt import matrix, solvers
    solvers.options['show_progress'] = verbose
    res = solvers.lp(matrix(c[0, :]), matrix(A_ub), matrix(b_ub))

    def normalize(vals):
        """
        Helper function to normalize a vector to the range (0, 1)
        """
        min_val = np.min(vals)
        max_val = np.max(vals)
        return (vals - min_val) / (max_val - min_val)
    
    # Extract the true optimisation variables and re-scale
    rewards = Rmax * normalize(res['x'][0:n]).T

    return rewards, res


def large_linear_programming(s0, k, T, phi, *, N=1, p=2.0, verbose=False):
    """Linear Programming IRL for large state spaces by Ng and Russell, 2000

    Given a sampling transition function T(s, a_i) -> s' encoding a stationary
    deterministic policy and a set of basis functions phi(s) over the state
    space, finds a weight vector alpha for which the given policy is optimal
    with respect to R(s) = alpha · phi(s).

    See https://thinkingwires.com/posts/2018-02-13-irl-tutorial-1.html for a
    good overview of this method.

    Args:
        s0 (list): A list of |s0|=M states that will be used to approximate the
            reward function over the full state space
        k (int): The number of actions (|A|)
        T (function): A sampling transition function T(s, a_i) -> s' encoding
            a stationary deterministic policy. The structure of T must be that
            the 0th action T(:, 0) corresponds to a sample from the expert
            policy, and T(:, i), i!=0 corresponds to a sample from the ith
            non-expert action at each state, for some arbitrary but consistent
            ordering of actions.
        phi (list of functions): A vector of basis functions phi_i(s) that
            take a state and give a float.

        N (int): Number of transition samples to use when computing
            expectations. For deterministic MDPs, this can be left as 1.
        p (float): Penalty function coefficient. Ng and Russell find p=2 is
            robust. Must be >= 1.
        verbose (bool): Print progress information

    Returns:
        (numpy array) A weight vector for the reward function basis functions
            that makes the given policy optimal
        (dict): A result object from the LP optimiser
    """

    # Measure number of sampled states
    M = len(s0)

    # Measure number of basis functions
    d = len(phi)

    # Enforce valid penalty function coefficient
    assert p >= 1, \
        "Penalty function coefficient must be >= 1, was {}".format(p)

    # Helper to estimate the mean (expectation) of a stochastic function
    E = lambda fn: sum([fn() for n in range(N)]) / N

    # Precompute the value expectation tensor VE
    # This is an array of shape (d, k-1, M) where VE[:, i, j] is a vector of
    # coefficients that, when multiplied with the alpha vector give the
    # expected difference in value between the expert policy action and the
    # ith action from state j
    VE_tensor = np.zeros(shape=(d, k - 1, M))

    # Loop over sampled initial states
    for j, s_j in enumerate(s0):
        if j % max(int(M / 20), 1) == 0 and verbose:
            print("Computing expectations... ({:.1f}%)".format(j / M * 100))

        # Compute E[phi(s')] where s' is drawn from the expert policy
        expert_basis_expectations = np.array([
            E(lambda: phi_i(T(s_j, 0))) for phi_i in phi
        ])

        # Loop over k-1 non-expert actions
        for i in range(1, k):
            # Compute E[phi(s')] where s' is drawn from the ith non-expert action
            ith_non_expert_basis_expectations = np.array([
                E(lambda: phi_i(T(s_j, i))) for phi_i in phi
            ])

            # Compute and store the expectation difference for this initial
            # state
            VE_tensor[:, i - 1, j] = expert_basis_expectations - \
                                     ith_non_expert_basis_expectations

    # TODO ajs 06/Jun/18 Remove redundant and trivial VE_tensor entries as
    # they create duplicate constraints

    # Formulate the linear programming problem constraints
    # NB: The general form for adding a constraint looks like this
    # c, A_ub, b_ub = f(c, A_ub, b_ub)
    if verbose: print("Composing LP problem...")

    def add_costly_single_step_constraints(c, A_ub, b_ub):
        """
        Augments the objective and adds constraints to implement the Linear
        Programming IRL method for large state spaces

        This will add up to M extra variables and 2M*(k-1) constraints
        (it does not add 'trivial' constraints)

        NB: Assumes the true optimisation variables are first in the c vector
        """

        # Step 1: Add the extra optimisation variables for each min{} operator
        # (one per sampled state)
        c = np.hstack([np.zeros(shape=(1, d)), np.ones(shape=(1, M))])
        A_ub = np.hstack([A_ub, np.zeros(shape=(A_ub.shape[0], M))])

        # Step 2: Add the constraints

        # Loop for each of the starting sampled states s_j
        for j in range(VE_tensor.shape[2]):
            if j % max(int(M / 20), 1) == 0 and verbose:
                print("Adding constraints... ({:.1f}%)".format(j / M * 100))

            # Loop over the k-1 non-expert actions
            for i in range(1, k):
                # Add two constraints, one for each half of the penalty
                # function p(x)
                constraint_row = np.hstack([VE_tensor[:, i - 1, j], \
                                            np.zeros(shape=M)])
                constraint_row[d + j] = -1
                A_ub = np.vstack((A_ub, constraint_row))
                b_ub = np.vstack((b_ub, 0))

                constraint_row = np.hstack([p * VE_tensor[:, i - 1, j], \
                                            np.zeros(shape=M)])
                constraint_row[d + j] = -1
                A_ub = np.vstack((A_ub, constraint_row))
                b_ub = np.vstack((b_ub, 0))

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

    # Prepare LP constraint matrices
    c = np.zeros(shape=[1, d], dtype=float)
    A_ub = np.zeros(shape=[0, d], dtype=float)
    b_ub = np.zeros(shape=[0, 1])

    # Compose LP optimisation problem
    c, A_ub, b_ub = add_costly_single_step_constraints(c, A_ub, b_ub)
    c, A_ub, b_ub = add_alpha_size_constraints(c, A_ub, b_ub)

    if verbose:
        print("Number of optimisation variables: {}".format(c.shape[1]))
        print("Number of constraints: {}".format(A_ub.shape[0]))

    # Solve for a solution
    if verbose: print("Solving LP problem...")

    # NB: cvxopt.solvers.lp expects a 1d c vector
    solvers.options['show_progress'] = verbose
    res = solvers.lp(matrix(c[0, :]), matrix(A_ub), matrix(b_ub))

    # Extract the true optimisation variables
    alpha_vector = res['x'][0:d].T

    return alpha_vector, res


def trajectory_linear_programming(
        zeta,
        T,
        S_bounds,
        A,
        phi,
        gamma,
        opt_pol,
        *,
        p=2.0,
        m=5000,
        H=30,
        tol=1e-6,
        verbose=False,
        on_iteration=None
):
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
            state_disc_index = round(
                (state - s_min) / (s_max - s_min) * (N - 1))
            state_disc_index = min(max(state_disc_index, 0), N - 1)
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

    # Demonstrate these methods on some gridworld problems
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid
    from irl_methods.utils import gaussian, indicator, rollout
    from irl_methods.mdp.gridworld import (
        GridWorldDiscEnv,
        GridWorldCtsEnv,
        EDGEMODE_WRAP,
        EDGEMODE_CLAMP
    )

    # ========= LP IRL

    # Construct a toy discrete gridworld with some randomized parameters to show
    # variance in the methods
    size = 8
    cell_size = 1 / size
    goal_state = np.random.randint(0, size, 2)
    gw_disc = GridWorldDiscEnv(
        size=size,
        goal_states=[goal_state],
        per_step_reward=0,
        goal_reward=1,
        edge_mode=EDGEMODE_CLAMP
    )
    disc_optimal_policy = gw_disc.get_optimal_policy()
    sorted_transition_tensor = gw_disc.ordered_transition_tensor(
        disc_optimal_policy
    )
    discount_factor = np.random.uniform(0, 1)
    l1 = np.random.uniform(0, 1)

    # Run LP IRL
    lp_reward, _ = linear_programming(
        sorted_transition_tensor,
        discount_factor,
        l1=l1,
        verbose=True
    )

    # ========= LLP IRL

    # Construct a toy continuous gridworld with some randomized parameters to
    # show variance in the methods
    initial_state = ((goal_state / size) + 0.5) % 1.0 +\
        np.random.uniform(0, 0.25, 2)
    step_size = np.random.uniform(0.001, 0.05)
    wind_size = np.random.uniform(0.01, 0.1)
    gw_cts = GridWorldCtsEnv(
        action_distance=step_size,
        wind_range=wind_size,
        initial_state=initial_state,
        goal_range=[goal_state / size, goal_state / size + cell_size],
        per_step_reward=0,
        goal_reward=1,
        edge_mode=EDGEMODE_CLAMP
    )
    cts_optimal_policy = gw_cts.get_optimal_policy()
    ordered_transition_function = gw_cts.ordered_transition_function(
        cts_optimal_policy
    )

    # Define a set of basis functions
    basis_function_means = [
        (x/size + cell_size/2, y/size + cell_size/2)
        for y in range(size)
        for x in range(size)
    ]
    sigma = 0.125
    covariance_matrix = np.diag((sigma, sigma))
    basis_functions = [
        gaussian(mu, covariance_matrix)
        for mu in basis_function_means
    ]
    # basis_functions = [
    #     indicator(mu, np.array((cell_size, cell_size)))
    #     for mu in basis_function_means
    # ]

    # Run LP IRL for large state spaces
    num_samples = 100
    state_sample = np.random.uniform(0, 1, (num_samples, 2))
    num_actions = len(gw_cts._A)
    num_transition_samples = 1
    penalty_coefficient = np.random.uniform(1, 5, 1)
    alpha_vector, _ = large_linear_programming(
        state_sample,
        num_actions,
        ordered_transition_function,
        basis_functions,
        verbose=True,
        N=num_transition_samples,
        p=penalty_coefficient
    )

    # Compose reward function lambda
    cts_reward = lambda s: np.dot(
        alpha_vector,
        [fn(s) for fn in basis_functions]
    )[0]

    # ========= TLP IRL

    # Roll-out some expert trajectories
    num_trajectories = num_samples
    max_trajectory_length = (0.5 / step_size) * 3
    trajectories = []
    for i in range(num_trajectories):
        trajectory, _ = rollout(
            gw_cts,
            initial_state,
            cts_optimal_policy,
            max_length=max_trajectory_length
        )
        trajectories.append(trajectory)

    # Make a new figure
    fig = plt.figure()
    plt.set_cmap("viridis")
    font_size = 7

    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(3, 3),
        axes_pad=0.3,
        share_all=False,
        label_mode="L",
        cbar_location="right",
        cbar_mode="single",
        cbar_size="5%",
        cbar_pad=0.15,
    )

    # ========= LP IRL

    # Plot ground truth reward
    plt.sca(grid[0])
    gw_disc.plot_reward(grid[0], gw_disc.ground_truth_reward, r_min=0, r_max=1)
    plt.title("Ground truth reward", fontsize=font_size)
    plt.yticks([])

    # Plot provided policy
    plt.sca(grid[1])
    gw_disc.plot_policy(grid[1], disc_optimal_policy)
    plt.title("Provided policy", fontsize=font_size)

    # Plot recovered reward
    plt.sca(grid[2])
    gw_disc.plot_reward(grid[2], lp_reward, r_min=0, r_max=1)
    plt.title("LP IRL result", fontsize=font_size)

    # ========= LP IRL for large state spaces

    # Plot ground truth reward
    plt.sca(grid[3])
    gw_cts.plot_reward(
        grid[3],
        gw_cts.ground_truth_reward,
        0,
        1,
        resolution=size
    )
    plt.title("Ground truth reward", fontsize=font_size)
    plt.yticks([])

    # Plot provided policy and state sample
    plt.sca(grid[4])
    gw_cts.plot_policy(
        grid[4],
        cts_optimal_policy,
        resolution=size
    )
    plt.scatter(
        state_sample[:, 0],
        state_sample[:, 1],
        s=10,
        alpha=0.3
    )
    plt.title("Provided policy", fontsize=font_size)

    # Plot recovered reward
    plt.sca(grid[5])
    gw_cts.plot_reward(
        grid[5],
        cts_reward,
        0,
        1,
        resolution=100
    )
    plt.title("LLP IRL result", fontsize=font_size)

    # ========= LP IRL with trajectories

    # Plot ground truth reward
    plt.sca(grid[6])
    gw_cts.plot_reward(
        grid[6],
        gw_cts.ground_truth_reward,
        0,
        1,
        resolution=size
    )
    plt.title("Ground truth reward", fontsize=font_size)
    plt.xticks([])
    plt.yticks([])

    # Plot provided trajectories
    plt.sca(grid[7])
    gw_cts.plot_trajectories(grid[7], trajectories)
    plt.plot(initial_state[0], initial_state[1], 'b.')
    plt.title("Provided trajectories", fontsize=font_size)
    plt.xticks([])

    # Plot recovered reward
    plt.sca(grid[8])
    #gw_cts.plot_policy(grid[8], cts_optimal_policy)
    plt.title("TLP IRL result", fontsize=font_size)
    plt.xticks([])

    # Add colorbar
    plt.sca(grid[0])
    plt.yticks([])
    plt.colorbar(cax=grid[0].cax)

    plt.show()



