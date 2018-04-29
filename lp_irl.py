"""
Implementation of Linear Programming IRL by Ng and Russell, 2000
(c) 2018 Aaron Snoswell
"""

import math
import copy
import numpy as np

from cvxopt import matrix, solvers


def lp_irl(T, gamma, l1, *, Rmax=1.0):
    """
    Implements Linear Programming IRL by NG and Russell, 2000

    Given a transition matrix T[s, a, s'] encoding a stationary, deterministic
    policy and a discount factor gamma finds a reward vector R(s) for which
    the policy is optimal.

    This method uses the Linear Programming IRL algorithm by Ng and Russell,
    2000 (http://ai.stanford.edu/~ang/papers/icml00-irl.pdf). See
    https://www.inf.ed.ac.uk/teaching/courses/rl/slides17/8_IRL.pdf for a more
    accessible overview.

    @param T - A sorted transition matrix T[s, a, s'] encoding a stationary
        deterministic policy. The structure of T must be that the 0th action
        T[:, 0, :] corresponds to the expert policy, and T[:, i, :], i != 0
        corresponds to the ith non-expert action at each state
    @param gamma - The expert's discount factor
    @param l1 - L1 regularization weight for LP optimisation objective
        function

    @param Rmax - Maximum reward value

    @return A reward vector for which the given policy is optimal
    @return A result object from the LP optimiser

    TODO: Adjust L1 norm constraint generation to allow negative rewards in
    the final vector. 
    """

    # Measure size of state and action sets
    n = T.shape[0]
    k = T.shape[1]

    # Compute the discounted transition matrix term
    T_disc_inv = np.linalg.inv(np.identity(n) - gamma * T[:, 0, :])

    # Formulate the linear programming problem constraints
    # NB: The general form for adding a constraint looks like this
    # c, A_ub, b_ub = f(c, A_ub, b_ub)

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

    # Solve for a solution

    # NB: cvxopt.solvers.lp expects a 1d c vector
    from cvxopt import matrix, solvers
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


if __name__ == "__main__":


    def build_sorted_transition_matrix(S, A, T, pi):
        """
        Given a vector of states S, a vector of actions A, a transition matrix
        T(s-a, s') and a policy dictionary, builds a sorted transition matrix
        T(s, a, s'), where the 0th action T(:, 0, :) corresponds to the expert
        policy, and the ith action T(:, i, :), i!=0 corresponds to the ith non
        -expert action at each state
        """

        # Build the compact form Transition matrix
        n = len(S)
        k = len(A)

        # Helper function to get a transition probability
        si = lambda s: S.tolist().index(s)
        ai = lambda a: A.tolist().index(a)
        trans = lambda s1, a, s2: T[si(s1) * k + ai(a), si(s2)]

        Tfull = np.zeros(shape=[n, k, n])
        for s_from in S:
            # Build vector of actions, sorted with the expert one first
            expert_action = pi[s_from]
            sorted_actions = np.append([expert_action], np.delete(A, ai(expert_action)))
            for a in sorted_actions:
                for s_to in S:
                    Tfull[si(s_from), ai(a), si(s_to)] = trans(s_from, a, s_to)

        return Tfull


    # Sample problem for lp_irl
    T = build_sorted_transition_matrix(
        np.array(["s0", "s1", "s2"]),
        np.array(["b", "o"]),
        np.array([[0,    0.4, 0.6 ],
                  [0,    0,   1   ],
                  [0,    0,   1   ],
                  [0,    0,   1   ],
                  [1,    0,   0   ],
                  [1,    0,   0   ]]),
        {
            "s0": "b",
            "s1": "o",
            "s2": "o"
        }
    )

    # Try LP IRL
    print(T)
    rewards, _ = lp_irl(T, 0.9, l1=10)
    print(rewards)
