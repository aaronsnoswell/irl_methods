# -*- coding: utf-8 -*-
"""Implementation of Linear Programming IRL methods by Ng and Russell, 2000

Copyright 2018 Aaron Snoswell
"""

import warnings
import numpy as np
from cvxopt import matrix, solvers
from pprint import pprint
from copy import copy

from irl_methods.utils import rollout


def linear_programming(
        sorted_transition_tensor,
        discount_factor,
        *,
        l1_regularisation_weight=0,
        r_max=1.0,
        verbose=False
):
    """Linear Programming IRL by NG and Russell, 2000

    Given a transition matrix T[s, a, s'] encoding a stationary
    deterministic policy and a discount factor, finds a reward vector
    R(s) for which the policy is optimal.

    This method implements the `Linear Programming IRL algorithm by Ng and
    Russell <http://ai.stanford.edu/~ang/papers/icml00-irl.pdf>. See 
    <https://www.inf.ed.ac.uk/teaching/courses/rl/slides17/8_IRL.pdf> for an
    accessible overview.

    TODO: Adjust L1 norm constraint generation to allow negative rewards in
    the final vector.

    Args:
        sorted_transition_tensor (numpy array): A sorted transition matrix
            T[s, a, s'] encoding a stationary deterministic policy. The
            structure must be such that the 0th action T[:, 0, :] corresponds to
            the expert policy, and T[:, i, :], i != 0 corresponds to the ith
            non-expert action at each state.
        discount_factor (float): The expert's discount factor. Must be in the
            range [0, 1).
        
        l1_regularisation_weight (float): L1 norm regularization weight for the
            LP optimisation objective function
        r_max (float): Maximum reward value
        verbose (bool): Print progress information
    
    Returns:
        (numpy array): The reward vector for which the given policy is
            optimal, and
        (dict): A result object from the LP optimiser
    """

    # Measure size of state and action sets
    n = sorted_transition_tensor.shape[0]
    k = sorted_transition_tensor.shape[1]

    if verbose:
        print("Vanilla Linear Programming IRL")
        print(
            "num_states={:d}, num_actions={:d}, discount_factor={:.3f}, "
            "l1_regularisation_weight={:.3f}, r_max={:.3f}".format(
                n,
                k,
                discount_factor,
                l1_regularisation_weight,
                r_max
            )
        )

    # Compute the discounted transition matrix inverse term
    _T_disc_inv = np.linalg.inv(
        np.identity(n) - discount_factor * sorted_transition_tensor[:, 0, :]
    )

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
            constraint_rows = -1 * \
                (
                    sorted_transition_tensor[:, 0, :] - \
                    sorted_transition_tensor[:, i, :]
                )\
                @ _T_disc_inv
            A_ub = np.vstack((A_ub, constraint_rows))
            b_ub = np.vstack(
                (b_ub, np.zeros(shape=[constraint_rows.shape[0], 1]))
            )
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
            constraint_rows = -1 * (sorted_transition_tensor[:, 0, :] - sorted_transition_tensor[:, i, :]) @ _T_disc_inv

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
        Add constraints for a maximum R value r_max
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
    c, A_ub, b_ub = add_rmax_constraints(c, A_ub, b_ub, r_max)
    c, A_ub, b_ub = add_l1norm_constraints(c, A_ub, b_ub, l1_regularisation_weight)

    if verbose:
        print("Number of optimisation variables: {}".format(c.shape[1]))
        print("Number of constraints: {}".format(A_ub.shape[0]))

    # Solve for a solution
    if verbose:
        print("Solving LP problem...")

    # NB: cvxopt.solvers.lp expects a 1d c vector
    solvers.options['show_progress'] = verbose
    res = solvers.lp(matrix(c[0, :]), matrix(A_ub), matrix(b_ub))

    if verbose:
        pprint(res)

    def normalize(vals):
        """
        Helper function to normalize a vector to the range (0, 1)
        """
        min_val = np.min(vals)
        max_val = np.max(vals)
        return (vals - min_val) / (max_val - min_val)
    
    # Extract the true optimisation variables and re-scale
    rewards = r_max * normalize(res['x'][0:n]).T

    return rewards, res


def add_alpha_size_constraints(c, a_ub, b_ub, num_basis_functions):
    """Add constraints for maximum reward coefficient values of 1

    This will add 2*d extra constraints, where d is the number of basis
    functions.

    NB: Assumes the true optimisation variables are first in the c vector.

    Args:
        c (numpy array): Objective function weights
        a_ub (numpy array): Upper bound constraint coefficient matrix
        b_ub (numpy array): Upper bound constraint RHS vector

    Returns
        (numpy array): Objective function weights
        (numpy array): Upper bound constraint coefficient matrix
        (numpy array): Upper bound constraint RHS vector
    """
    for i in range(num_basis_functions):
        constraint_row = [0] * a_ub.shape[1]
        constraint_row[i] = 1
        a_ub = np.vstack((a_ub, constraint_row))
        b_ub = np.vstack((b_ub, 1))

        constraint_row = [0] * a_ub.shape[1]
        constraint_row[i] = -1
        a_ub = np.vstack((a_ub, constraint_row))
        b_ub = np.vstack((b_ub, 1))

    return c, a_ub, b_ub


def large_linear_programming(
    state_sample,
    num_actions,
    ordered_transition_function,
    basis_value_fn,
    *,
    penalty_coefficient=2.0,
    num_transition_samples=100,
    verbose=False
):
    """Linear Programming IRL for large state spaces by Ng and Russell, 2000

    Given a sampling transition function T(s, a_i) -> s' encoding a stationary
    deterministic policy and a set of basis functions phi(s) over the state
    space, finds a weight vector alpha for which the given policy is optimal
    with respect to R(s) = alpha · phi(s).

    See https://thinkingwires.com/posts/2018-02-13-irl-tutorial-1.html for a
    good overview of this method.

    Args:
        state_sample (list): A list of states that will be used to approximate
            the reward function over the full state space
        num_actions (int): The number of actions
        ordered_transition_function (function): A sampling transition function
            T(s, a_i) -> s' encoding a stationary deterministic policy. The
            structure of T must be that the 0th action T(:, 0) corresponds to a
            sample from the expert policy, and T(:, i), i!=0 corresponds to a
            sample from the ith non-expert action at each state, for some
            arbitrary but consistent ordering of actions.
        basis_value_fn (function): A function bv(s) -> list taking a state
            and returning a vector where each entry i is an estimate of the
            value of that state, if ith basis function was the reward function.

        penalty_coefficient (float): Penalty function coefficient. Ng and
            Russell find 2 is robust. Must be >= 1.
        num_transition_samples (int): Number of transition samples to use when
            computing expectations. For deterministic MDPs, this can be left
            as 1.
        verbose (bool): Print progress information

    Returns:
        (numpy array) A weight vector for the reward function basis functions
            that makes the given policy optimal
        (dict): A result object from the LP optimiser
    """

    # Enforce valid penalty function coefficient
    assert penalty_coefficient >= 1, \
        "Penalty function coefficient must be >= 1, was {}".format(
            penalty_coefficient
        )

    # Measure number of sampled states
    num_states = len(state_sample)

    # Measure number of basis functions
    num_basis_functions = len(basis_value_fn(state_sample[0]))

    if verbose:
        print("Large Linear Programming IRL")
        print(
            "num_states={}, num_actions={}, num_basis_functions={}, "
            "penalty_coefficient={:.3f}, num_transition_samples={}".format(
                num_states,
                num_actions,
                num_basis_functions,
                penalty_coefficient,
                num_transition_samples
            )
        )

    # Formulate the linear programming problem constraints
    # NB: The general form for adding a constraint looks like this
    # c, A_ub, b_ub = f(c, a_ub, b_ub)
    if verbose:
        print("Composing LP problem...")

    def add_costly_single_step_constraints(c, a_ub, b_ub):
        """Implement Linear Programming IRL method for large state spaces

        This will add m new objective parameters and 2*(k-1)*m new constraints,
        where k is the number of actions and m is the number of sampled states

        NB: Assumes the true optimisation variables are first in the c vector.

        Args:
            c (numpy array): Objective function weights
            a_ub (numpy array): Upper bound constraint coefficient matrix
            b_ub (numpy array): Upper bound constraint RHS vector

        Returns
            (numpy array): Objective function weights
            (numpy array): Upper bound constraint coefficient matrix
            (numpy array): Upper bound constraint RHS vector
        """

        # Extend the objective function adding one dummy parameter per
        # sampled state
        # NB: cvxopt minimises, so we have a -1 here, not a 1
        c = np.hstack((c, -1 * np.ones(shape=(1, num_states))))
        a_ub = np.hstack((a_ub, np.zeros(shape=(a_ub.shape[0], num_states))))

        # Loop over states
        for si, state in enumerate(state_sample):

            # Show progress
            if verbose:
                percent = int(si / num_states * 100)
                if percent % 5 == 0:
                    print("{:d}%".format(percent))

            # Compute the value-expectations for the possible actions
            a_ve = np.zeros((num_basis_functions, num_actions))
            for i in range(num_transition_samples):
                for a in range(num_actions):
                    a_ve[:, a] += basis_value_fn(
                        ordered_transition_function(state, a)
                    )
            a_ve /= num_transition_samples

            # Find the difference w.r.t the expert (0th action)
            a_ve_diff = np.subtract(np.array([a_ve[:, 0]]).T, a_ve[:, 1:])

            # Prepare the RHS block of the a_ub matrix
            tmp = np.zeros((1, num_states))
            tmp[0, si] = -1
            tmp = np.vstack((tmp for _ in range(num_actions-1)))

            # Append to first half of penalty constraints to a_ub, b_ub
            a_ub = np.vstack((
                a_ub,
                np.hstack((a_ve_diff.T, tmp))
            ))
            b_ub = np.vstack((
                b_ub,
                np.vstack((0 for _ in range(num_actions-1)))
            ))

            # Append to second half of penalty constraints to a_ub, b_ub
            a_ub = np.vstack((
                a_ub,
                np.hstack((penalty_coefficient * a_ve_diff.T, tmp))
            ))
            b_ub = np.vstack((
                b_ub,
                np.vstack((0 for _ in range(num_actions-1)))
            ))

        # TODO ajs 12/Jun/2018 Remove redundant constraints

        # *twitch*
        if verbose:
            print("100%")

        return c, a_ub, b_ub

    # Prepare LP constraint matrices
    c = np.zeros(shape=(1, num_basis_functions), dtype=float)
    a_ub = np.zeros(shape=(0, num_basis_functions), dtype=float)
    b_ub = np.zeros(shape=(0, 1))

    # Compose LP optimisation problem
    c, a_ub, b_ub = add_costly_single_step_constraints(c, a_ub, b_ub)
    c, a_ub, b_ub = add_alpha_size_constraints(c, a_ub, b_ub)

    if verbose:
        print("Number of optimisation variables: {}".format(c.shape[1]))
        print("Number of constraints: {}".format(a_ub.shape[0]))
        print("Solving LP problem...")

    # NB: cvxopt.solvers.lp expects a 1d c vector
    solvers.options['show_progress'] = verbose
    res = solvers.lp(matrix(c[0, :]), matrix(a_ub), matrix(b_ub))

    if verbose:
        pprint(res)

    # Extract the true optimisation variables
    alpha_vector = np.array(res['x'][0:num_basis_functions].T)

    return alpha_vector, res


def trajectory_linear_programming(
    mdp,
    trajectories,
    basis_functions,
    discount_factor,
    num_iterations,
    solver,
    *,
    penalty_coefficient=2.0,
    verbose=True
):
    """Trajectory-based Linear Programming IRL by Ng and Russell, 2000

    Args:
        mdp (gym.Env): MDP to run IRL over. Used for rollint out trajectories.
        trajectories (list): List of state trajectories from the expert
        basis_functions (list): List of basis functions b(s) -> float
        discount_factor (float): Expert's discount factor
        num_iterations (int): Number of iterations to loop for
        solver (function): A function that solves an mdp to find an optimal
            policy. Should take a gym.Env object and a reward function,
            and return a policy function pi(s) -> a mapping states to actions

        penalty_coefficient (float): Penalty function coefficient. Ng and
            Russell find 2 is robust. Must be >= 1.
        verbose (bool): Print status information

    Returns:
        (numpy array): Vector of basis function coefficients such that the
            expert's demonstrations are optimal.
    """

    # Enforce valid penalty function coefficient
    assert penalty_coefficient >= 1, \
        "Penalty function coefficient must be >= 1, was {}".format(
            penalty_coefficient
        )

    # Convenience variable for the number of basis functions
    d = len(basis_functions)

    # Slice out the initial state set
    starting_state_set = [t[0] for t in trajectories]

    # Measure maximum trajectory length
    max_trajectory_length = max([len(t) for t in trajectories])

    if verbose:
        print("Trajectory-based Linear Programming IRL")
        print(
            "num_trajectories={}, max_trajectory_length={}, "
            "num_basis_functions={}, "
            "penalty_coefficient={:.3f}, num_iterations={}".format(
                len(trajectories),
                max_trajectory_length,
                d,
                penalty_coefficient,
                num_iterations
            )
        )

    def generate_state_trajectories(
            mdp,
            start_state_set,
            policy,
            max_trajectory_length
    ):
        """Generate a list of state-only trajectories

        Args:
            mdp (gym.Env): The MDP to roll-out trajectories in
            start_state_set (list): List of starting states to use
            policy (function): Policy to follow for rollouts
            max_trajectory_length: Maximum trajectory length to allow

        Returns:
            (list): A list of state trajectories, (each a list of states)
        """

        trajectories = []
        for s0 in start_state_set:
            trajectory = rollout(
                mdp,
                s0,
                policy,
                max_length=max_trajectory_length
            )

            # Slice out just the state trajectory and append it
            trajectories.append([sar[0] for sar in trajectory])

        return trajectories

    def discounted_feature_expectations(
            trajectories,
            basis_functions,
            discount_factor
    ):
        """Compute discounted feature expectations under the basis function set

        Args:
            trajectories (list): List of state trajectories
            basis_functions (list): List of basis functions b(s) -> float
            discount_factor: Expert's discount factor

        Returns:
            (numpy array): A numpy array of shape (1, d) where d is the number
                of basis functions. The ith column indicates the average
                discounted feature expectation for feature i given the set of
                trajectories.
        """

        feature_expectations = np.zeros((1, len(basis_functions)))

        # Average over all trajectories
        for trajectory in trajectories:

            # Compute one row per basis function
            for i, basis_function in enumerate(basis_functions):
                discount_vector = np.array(
                    [discount_factor ** j for j in range(len(trajectory))]
                )
                feature_values = np.array(
                    [basis_function(s) for s in trajectory])

                feature_expectations[0, i] += discount_vector @ feature_values

        # Normalize expectations
        feature_expectations /= len(trajectories)

        return feature_expectations

    # Compute expert discounted feature expectations
    v_hat_pi_star = discounted_feature_expectations(
        trajectories,
        basis_functions,
        discount_factor
    )

    # Keep a list of discounted policy feature expectations
    policy_set = []
    policy_discounted_feature_expectations = np.empty(
        shape=(0, v_hat_pi_star.shape[1])
    )

    def add_optimal_expert_constraints(c, a_ub, b_ub):
        """Add constraints and augment the objective to make the expert optimal

        This will add k extra optimisation variables, and 2k extra
        constraints, where k is the number of policies in the policy set.

        NB: Assumes the true optimisation variables are first in the c vector.

        Args:
            c (numpy array): Objective function weights
            a_ub (numpy array): Upper bound constraint coefficient matrix
            b_ub (numpy array): Upper bound constraint RHS vector

        Returns
            (numpy array): Objective function weights
            (numpy array): Upper bound constraint coefficient matrix
            (numpy array): Upper bound constraint RHS vector
        """

        # Measure the current number of policies in the policy set
        k = policy_discounted_feature_expectations.shape[0]

        # Add optimisation variables for all policies
        c = np.hstack((c, np.ones((1, k))))
        a_ub = np.hstack((a_ub, np.zeros((a_ub.shape[0], k))))

        diff = v_hat_pi_star - policy_discounted_feature_expectations
        tmp = -1 * np.identity(policy_discounted_feature_expectations.shape[0])

        # Add Constraints for both sides of the penalty function
        a_ub = np.vstack((
            a_ub,
            np.hstack((diff, tmp)),
            np.hstack((penalty_coefficient * diff, tmp))
        ))
        b_ub = np.vstack((
            b_ub,
            np.zeros((k, 1)),
            np.zeros((k, 1))
        ))

        return c, a_ub, b_ub

    # Iterate the requested number of times
    alpha_vector = None
    for i in range(num_iterations):

        if alpha_vector is None:

            # First iteration - pick an initial, random policy
            policy_set.append(
                lambda s: mdp.action_space.sample()
            )

        else:

            # Make a copy of the MPD
            mdp_copy = copy(mdp)
            mdp_copy.reset()

            # Compose a new reward function
            new_reward_fn = lambda s: (
                    alpha_vector @ np.array([bfn(s) for bfn in basis_functions])
            )[0]

            def _step(self, action):
                """Overload an MDP's reward function

                Args:
                    action (object): an action provided by the environment

                Returns:
                    observation (object): agent's observation of the current
                        environment
                    reward (float) : amount of reward returned after previous
                        action
                    done (boolean): whether the episode has ended, in which
                        case further step() calls will return undefined results
                    info (dict): contains auxiliary diagnostic information
                        (helpful for debugging, and sometimes learning)
                """

                # Call original function
                s, r, d, i = self._step()

                # Replace reward
                r = new_reward_fn(s)

                return s, r, d, i

            # Overload the MDP to use a new reward function
            mdp_copy._step = mdp_copy.step
            mdp_copy.step = _step

            # Call the provided solver to find a policy that maximises V under
            # the new reward function
            policy_set.append(solver(mdp, new_reward_fn))

        # Estimate the value of the newest policy
        policy_discounted_feature_expectations = np.vstack((
            policy_discounted_feature_expectations,
            discounted_feature_expectations(
                generate_state_trajectories(
                    mdp,
                    starting_state_set,
                    policy_set[-1],
                    max_trajectory_length
                ),
                basis_functions,
                discount_factor
            )
        ))

        # Form the LP problem
        c = np.zeros((1, d), dtype=float)
        a_ub = np.zeros((0, c.shape[1]), dtype=float)
        b_ub = np.zeros((a_ub.shape[0], 1), dtype=float)

        c, a_ub, b_ub = add_optimal_expert_constraints(c, a_ub, b_ub)
        c, a_ub, b_ub = add_alpha_size_constraints(c, a_ub, b_ub, d)

        if verbose:
            print("Number of optimisation variables: {}".format(c.shape[1]))
            print("Number of constraints: {}".format(a_ub.shape[0]))
            print("Solving LP problem...")

        # NB: cvxopt.solvers.lp expects a 1d c vector
        solvers.options['show_progress'] = verbose
        res = solvers.lp(matrix(c[0, :]), matrix(a_ub), matrix(b_ub))

        if verbose:
            pprint(res)

        # Extract the true optimisation variables
        alpha_vector = np.array(res['x'][0:d].T)

        print(alpha_vector)

    return alpha_vector


def demo():
    """ Demonstrate these methods on some gridworld problems
    """

    # region === Get some imports
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from irl_methods.utils import gaussian
    from irl_methods.mdp.gridworld import (
        GridWorldDiscEnv,
        GridWorldCtsEnv,
        FEATUREMAP_IDENTITY,
        EDGEMODE_WRAP,
        EDGEMODE_CLAMP,
        EDGEMODE_STRINGS,
        ACTION_STRINGS
    )

    #endregion

    # region === GridWorld construction
    # #################################################################

    # Size of the gridworld
    size = 8

    # Wind probability for the discrete gridworld
    wind_prob = 0.3

    # Edge mode for the gridworlds
    edge_mode = EDGEMODE_CLAMP
    #edge_mode = EDGEMODE_WRAP

    # Per-step reward
    per_step_reward = 0

    # Goal reward
    goal_reward = 1

    # Goal state
    #goal_state = np.random.randint(0, size, 2)
    goal_state = np.array((size-1, size-1))

    # Discount factor
    #discount_factor = np.random.uniform(0.1, 0.9)
    discount_factor = 0.9

    print("Discrete GridWorld, "
        "size={:d}, wind_prob={:d}%, edge_mode={}".format(
            size,
            int(wind_prob*100),
            EDGEMODE_STRINGS[edge_mode]
        )
    )
    num_states = size * size
    gw_disc = GridWorldDiscEnv(
        size=size,
        wind=wind_prob,
        goal_states=[goal_state],
        per_step_reward=per_step_reward,
        goal_reward=goal_reward,
        edge_mode=edge_mode
    )
    disc_optimal_policy = gw_disc.get_optimal_policy()
    ordered_transition_tensor = gw_disc.get_ordered_transition_tensor(
        disc_optimal_policy
    )

    #endregion
    #
    # # region === Vanilla LP IRL
    # ##################################################################
    # print("")
    #
    # # L1 regularisation weight
    # # l1 = 0
    # l1 = np.random.uniform(0, 1)
    #
    # # ===
    #
    # # Run LP IRL
    # lp_reward, _ = linear_programming(
    #     ordered_transition_tensor,
    #     discount_factor,
    #     l1_regularisation_weight=l1,
    #     r_max=(per_step_reward + goal_reward),
    #     verbose=True
    # )
    #
    # print("Recovered reward vector:")
    # print("{}".format(lp_reward))
    #
    # fig = plt.figure()
    # plt.suptitle('Vanilla Linear Programming IRL')
    # plt.set_cmap("viridis")
    #
    # # Plot ground truth reward
    # ax = plt.subplot(1, 3, 1)
    # gw_disc.plot_reward(ax, gw_disc.ground_truth_reward)
    # plt.title("Ground truth reward")
    # plt.colorbar(
    #     cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    # )
    #
    # # Plot provided policy
    # ax = plt.subplot(1, 3, 2)
    # gw_disc.plot_policy(ax, disc_optimal_policy)
    # plt.title("Provided policy")
    #
    # # Plot recovered reward
    # ax = plt.subplot(1, 3, 3)
    # gw_disc.plot_reward(ax, lp_reward)
    # plt.title("IRL result")
    # plt.colorbar(
    #     cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    # )
    #
    # plt.tight_layout()
    # plt.show()
    #
    # # endregion
    #
    # # region === LLP IRL
    # ##################################################################
    #
    # # Number of transition samples to use when estimating values in LLP
    # # If the wind is 0, this can be 1
    # num_transition_samples = 100
    #
    # # Penalty coefficient to use in LLP/TLP
    # # Min value is 1, Ng and Russell suggest 2
    # penalty_coefficient = 10
    #
    # # ===
    #
    # # Feature function
    # feature_fn = lambda s: gw_disc.get_state_features(
    #     s=s,
    #     feature_map=FEATUREMAP_IDENTITY
    # )
    #
    # # Convert basis functions to value functions
    # print("Computing value estimates...")
    # basis_value_functions = []
    # for bfi in range(num_states):
    #
    #     # Progress update
    #     print("{:d}%".format(int(bfi/num_states*100)))
    #
    #     basis_value_functions.append(
    #         gw_disc.estimate_value(
    #             disc_optimal_policy,
    #             discount_factor,
    #             reward=lambda s: feature_fn(s)[bfi]
    #         )
    #     )
    # # A vector function that computes the value vector given a state
    # basis_value_fn = lambda s: [bvf[s] for bvf in basis_value_functions]
    # print("")
    #
    # state_sample = list(range(num_states))
    # num_actions = len(gw_disc._A)
    # ordered_transition_function = lambda s, a: np.random.choice(
    #     state_sample,
    #     p=ordered_transition_tensor[s, a]
    # )
    #
    # alpha_vector, _ = large_linear_programming(
    #     state_sample,
    #     num_actions,
    #     ordered_transition_function,
    #     basis_value_fn,
    #     penalty_coefficient=penalty_coefficient,
    #     num_transition_samples=num_transition_samples,
    #     verbose=True
    # )
    #
    # print("Recovered alpha vector:")
    # print("{}".format(alpha_vector))
    #
    # # Compose reward function lambda
    # llp_reward_fn = lambda s: (alpha_vector @ basis_value_fn(s))[0]
    # llp_reward_vector = [llp_reward_fn(s) for s in range(num_states)]
    #
    # # Plot results
    # fig = plt.figure()
    # plt.suptitle('Large Linear Programming IRL')
    # plt.set_cmap("viridis")
    #
    # # Plot ground truth reward
    # ax = plt.subplot(1, 3, 1)
    # gw_disc.plot_reward(ax, gw_disc.ground_truth_reward)
    # plt.title("Ground truth reward")
    # plt.colorbar(
    #     cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    # )
    #
    # # Plot provided policy
    # ax = plt.subplot(1, 3, 2)
    # gw_disc.plot_policy(ax, disc_optimal_policy)
    # plt.title("Provided policy")
    #
    # # Plot recovered reward
    # ax = plt.subplot(1, 3, 3)
    # gw_disc.plot_reward(ax, llp_reward_vector)
    # plt.title("IRL result")
    # plt.colorbar(
    #     cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    # )
    #
    # plt.tight_layout()
    # plt.show()
    #
    # # endregion

    # region === TLP IRL
    ##################################################################

    # Maximum trajectory length to use for TLP
    max_trajectory_length = 3 * size

    # Number of iterations to run the TLP solver for
    num_iterations = 20

    # Number demonstration trajectories to provide to the TLP algorithm
    num_trajectories = 5000

    # Roll out some trajectories to use for TLP
    trajectories = []
    for i in range(num_trajectories):
        trajectory = rollout(
            gw_disc,
            gw_disc.observation_space.sample(),
            disc_optimal_policy,
            max_length=max_trajectory_length
        )

        # Slice out just the state trajectory and append it
        trajectories.append([sar[0] for sar in trajectory])

    # A vector of basis functions
    basis_functions = [
        lambda s: gw_disc.get_state_features(s=s)[0],
        lambda s: gw_disc.get_state_features(s=s)[1]
    ]

    def solver(mdp, reward):
        """Solves a GridWorldDiscEnv MDP using policy iteration to find a policy

        Args:
            mdp (GridWorldDiscEnv): MDP with a valid reward function to solve
            reward (function): Reward function r(s) -> float

        Returns:
            (function): Policy pi(s) -> a that acts optimally with respect to
                the given MDP's reward function
        """

        # Convergence tolerance for the value function
        tol = 1e-6

        # Max number of PI iterations
        max_num_iterations = 10

        # Start with random policy and zero value vector
        pi = lambda s: mdp.action_space.sample()
        v = np.zeros(len(mdp._S))

        print("Running PI")

        i = 0
        while True:
            i += 1

            print("PI step {}".format(i))

            # Update value
            v_new = mdp.estimate_value(
                pi,
                discount_factor,
                reward=reward,
                max_iterations=1000
            )
            delta = v_new - v
            v = v_new
            print("PI: Value delta is: {}".format(np.linalg.norm(delta)))

            # Get new greedy policy
            pi = mdp.greedy_policy(v_new)

            if np.linalg.norm(delta) < tol or i >= max_num_iterations:
                break

        print("Got new policy!!!")

        return pi

    alpha_vector = trajectory_linear_programming(
        gw_disc,
        trajectories,
        basis_functions,
        discount_factor,
        num_iterations,
        solver,
        verbose=True
    )

    tlp_reward_fn = lambda s: (alpha_vector @ [bf(s) for bf in basis_functions])[0]
    tlp_reward_vector = [tlp_reward_fn(s) for s in range(num_states)]

    # Plot results
    fig = plt.figure()
    plt.suptitle('Trajectory-based Linear Programming IRL')
    plt.set_cmap("viridis")

    # Plot ground truth reward
    ax = plt.subplot(1, 3, 1)
    gw_disc.plot_reward(ax, gw_disc.ground_truth_reward)
    plt.title("Ground truth reward")
    plt.colorbar(
        cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    )

    # Plot provided policy
    #ax = plt.subplot(1, 3, 2)
    #gw_disc.plot_policy(ax, disc_optimal_policy)
    plt.title("Provided trajectories")

    # Plot recovered reward
    ax = plt.subplot(1, 3, 3)
    gw_disc.plot_reward(ax, tlp_reward_vector)
    plt.title("IRL result")
    plt.colorbar(
        cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    )

    plt.tight_layout()
    plt.show()

    # endregion

    return


if __name__ == "__main__":
    demo()
