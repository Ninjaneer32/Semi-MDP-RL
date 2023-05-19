from typing import List
import numpy as np
import gurobipy as gp
from gurobipy import GRB


class MPCController:
    """
    Implementation of a MPC controller for the lifter control problem.
    In each time step, mixed-integer bilinear program is defined based on the system state and is solved using Gurobi.
    """
    def __init__(self,
                 actions,
                 stay_action,
                 points,
                 num_steps,
                 horizon_length,
                 conveyors,
                 capacities,
                 reward_vector,
                 time_matrix,
                 position_matrix,
                 floors,
                 L_u, L_ell, U_u, U_ell
                 ):
        # len(capacities) == conveyors
        self._actions = actions
        
        self._points = points
        
        # number of time steps considered in each MPC problem
        self._K = num_steps
        self._horizon = horizon_length
        self._floors = floors
        self._conveyors = conveyors
        self._capacities = capacities

        self._r_vec = reward_vector

        # time_matrix.shape == (points, actions)
        self._t_mat = time_matrix
        self._p_mat = position_matrix
        self._params = None

        self._stay = stay_action

        self._arrival_mat = {}

        self._L_u = L_u
        self._L_ell = L_ell

        self._L = {c: {} for c in conveyors}
        for c in conveyors:
            for a in actions:
                self._L[c][a] = self._L_u[c][a] + self._L_ell[c][a]

        self._U_u = U_u
        self._U_ell = U_ell
        
        # not used in the current version
        self._discount = .9


        return

    def update_model_params(self, params):
        """
        Update the estimated parameters of the lot arrival processes, which are modeled as Poisson process here.
        The parameters are used when building the constraints for the mixed-integer bilinear programming problem.
        """
        self._params = params
        return

    def act(self, lifter_pos: int, destinations: List[int], upper: int, lower: int, num_lots: List[int]):
        """
        Computation of a single control input (or action) given the state of the fab.
        When called, the mixed-integer bi-linear programming problem is formulated & solved using Gurobi.
        
        The following arguments describe the state of the lifter control sytem of fab:
        -------------------------------------------------------------------------------------------------
        :param lifter_pos: position of the lifter
        
        :param destinations: destinations of the foremost lots in the layers
        
        :param upper: destination of the lot at upper fork of the lifter
        
        :param lower: destination of the lot at lower fork of the lifter
        
        :param num_lots: number of lots waiting in the layers
        -------------------------------------------------------------------------------------------------
        return: an action chosen from the set of admissible actions
        """
        assert upper == 0 or upper in self._floors
        assert lower == 0 or lower in self._floors

        assert lifter_pos in self._points

        # problem construction using Gurobi
        m = gp.Model('mpc')
        m.Params.LogToConsole = 0
        m.params.NonConvex = 2
        # binary vector encoding of the actions taken in decision epochs
        # Each action a_i is encoded into a one-hot vector whose i-th entry is 1.
        A = m.addVars(self._actions, self._K, vtype=GRB.INTEGER, name='A')
        m.addConstrs(constrs=(sum(A[a, k] for a in self._actions) == 1 for k in range(self._K)),
                     name='action_embedding')

        # position matrix
        for p in self._points:
            self._p_mat[p][self._stay] = 0
        self._p_mat[lifter_pos][self._stay] = 1

        # positions P_k of the lifter encoded as binary vectors
        P = m.addVars(self._points, self._K + 1, vtype=GRB.INTEGER, name='pos')
        m.addConstr(P[lifter_pos, 0] == 1, name='P')
        m.addConstrs(constrs=(sum(P[p, k] for p in self._points) == 1 for k in range(self._K + 1)),
                     name='pos_embedding')

        
        """
        N: vector representing the number of waiting lots in the layers
        The equation describing the changes in the number of waiting lots is given as follows:

        N^c_{k+1} = min{N^c_k + tau_{P_k, A_k} lambda^c - L^c_{A_k}, bar{N}^c}

        where
        
        -------------------------------------------------------------------------------------------------
        c: conveyor & C: set of all conveyors
        
        tau_{P_k, A_k}: execution time of action A_k at position P_k
        
        lambda^c: estimated lot arrival frequency of conveyor c
        
        L^c_{A_k}: binary variable representing whether A_k loads a lot from each c
        
        bar{N}^c: capacity of conveyor c
        -------------------------------------------------------------------------------------------------
        """
        N = m.addVars(self._conveyors, self._K + 1, vtype=GRB.CONTINUOUS, name='N')
        """
        D^u/D^ell: binary vector of length |F| representing the destination of the lot loaded at upper/lower fork
                   where F : set of all target floors
        
        D^u = 0 if no lot is at upper fork, same for D^ell
        """
        D_u = m.addVars(self._floors, self._K + 1, vtype=GRB.INTEGER, name='D^upper')
        D_ell = m.addVars(self._floors, self._K + 1, vtype=GRB.INTEGER, name='D^lower')

        m.addConstrs(constrs=(sum(D_u[f, k] for f in self._floors) <= 1 for k in range(self._K + 1)),
                     name='D^upper_cap')
        m.addConstrs(constrs=(sum(D_ell[f, k] for f in self._floors) <= 1 for k in range(self._K + 1)),
                     name='D^lower_cap')
        # TODO : initialize
        if upper == 0:
            m.addConstrs((D_u[f, 0] == 0 for f in self._floors), name='D0^upper')
        else:
            m.addConstrs((D_u[f, 0] == 0 for f in self._floors if f != upper), name='D0^upper')
            m.addConstr(D_u[upper, 0] == 1, name='D0^upper_filled')

        if lower == 0:
            m.addConstrs((D_ell[f, 0] == 0 for f in self._floors), name='D0^lower')
        else:
            m.addConstrs((D_ell[f, 0] == 0 for f in self._floors if f != lower), name='D0^lower_fork')
            m.addConstr(D_ell[lower, 0] == 1, name='D0^lower_filled')

        m.addConstrs((N[c, 0] == num_lots[c] for c in self._conveyors), name='N0')

        ts = []
        for k in range(self._K):
            """
            equivalent expression for bilinear product using new matrix variable Q_k
            The product x * y of two binary variables can be written as a new binary (but okay to set continuous) variable z which satisfies
            z >= 0,
            z >= x + y - 1,
            z <= x,
            z <= y.
            """
            Qk = m.addVars(self._actions, self._points, vtype=GRB.CONTINUOUS, name='Q{}'.format(k))
            m.addConstrs((Qk[a, p] <= A[a, k] for a in self._actions for p in self._points),
                         name='QA{}'.format(k))
            m.addConstrs((Qk[a, p] <= P[p, k] for a in self._actions for p in self._points),
                         name='QP{}'.format(k))
            m.addConstrs(
                (Qk[a, p] >= A[a, k] + P[p, k] - 1 for a in self._actions for p in self._points),
                name='QAP{}'.format(k))

            """
            new variables S_k to handle the capacity of the conveyor belts
            Briefly speaking, the equation of the form z = min{x, y} is transformed to the linear constraints
            z <= x,
            z <= y,
            z >= x + (l2 - u1) * s,
            z >= y + (l1 - u2) * (1 - s),
            by introducing a new binary variable s. (Here we assume that the bounds l1 <= x <= u1 & l2 <= y <= u2 are known.)
            """
            S = m.addVars(self._conveyors, vtype=GRB.BINARY, name='S{}'.format(k))
            # execution time of the action at step k
            tk = m.addVar(lb=0., ub=float('inf'), obj=0., vtype=GRB.CONTINUOUS, name='t{}'.format(k), column=None)
            m.addConstr(
                tk == sum(self._t_mat[p][a] * Qk[a, p] for a in self._actions for p in self._points),
                name='time definition{}'.format(k))
            ts.append(tk)

            # modified lot arrival equation using new variables S_k
            m.addConstrs((N[c, k + 1] <= N[c, k] + tk * self._params[c] - sum(
                self._L[c][a] * A[a, k] for a in self._actions) for c in self._conveyors),
                name='N-ineq1-{}'.format(k))
            m.addConstrs((N[c, k + 1] <= self._capacities[c] for c in self._conveyors),
                    name='N-ineq2-{}'.format(k))
            m.addConstrs((N[c, k + 1] >= self._capacities[c] * S[c] for c in self._conveyors), name='N-ineq3-{}'.format(k))
            m.addConstrs((N[c, k + 1] >= N[c, k] + tk * self._params[c] - sum(
                self._L[c][a] * A[a, k] for a in self._actions) - self._t_max * self._lambda_max * S[c]
                for c in self._conveyors), name='N-ineq4-{}'.format(k))

            # constraint N^c_k >= L^c_{A_k}: in order to load a lot from c, there must be at least 1 lot in c. 
            m.addConstrs((N[c, k] >= sum(self._L[c][a] * A[a, k] for a in self._actions) for c in
                          self._conveyors), name='N-lb{}'.format(k))

            """
            The equations describing the state of the lifters are as follows:

            D^u_{k+1} = D^u_k + DD_k @ L^u_{A_k} - U^u_{A_k},
            D^ell_{k+1} = D^ell_k + DD_k @ L^ell_{A_k} - U^ell_{A_k},
            
            where
            
            -------------------------------------------------------------------------------------------------
            DD_k: randomly generated binary matrix denoting the destination of the foremost lot in each layer
                  Specifically, DD_k[f, c] = 1 iff f is the destination of the foremost lot in c.

            L^u_{A_k}: binary vector of length |C| indicating whether A_k loads a lot from upper fork.
                       The vector L_{A_k} defined above is the same as L^u_{A_k} + L^ell_{A_k}.
            
            U^u_{A_k}: binary variable representing whether A_k unloads a lot from upper fork
            
            U^ell_{A_k}: binary variable representing whether A_k unloads a lot from lower fork
            -------------------------------------------------------------------------------------------------
            """
            m.addConstrs((D_u[f, k + 1] == D_u[f, k] + sum(
                (sum(destinations[f][c][k] * self._L_u[c][a] for c in self._conveyors) - self._U_u[f][a]) * A[
                    a, k] for a in self._actions) for f in self._floors),
                         name='upper_fork{}'.format(k))
            m.addConstrs((D_ell[f, k + 1] == D_ell[f, k] + sum(
                (sum(destinations[f][c][k] * self._L_ell[c][a] for c in self._conveyors) - self._U_ell[f][a]) *
                A[a, k] for a in self._actions) for f in self._floors), name='lower_fork{}'.format(k))
            # equation describing the position of the lifter
            m.addConstrs((P[p, k + 1] == sum(self._p_mat[p][a] * A[a, k] for a in self._actions) for p in
                          self._points), name='P-eq{}'.format(k))


        # time constraint: total execution time for next K steps must be less than equal to T.
        m.addConstr(sum(t for t in ts) <= self._horizon, name='total_time')
        
        # The objective of the problem is the sum of linear functions of the action vectors A_k's.
        obj = sum(self._r_vec[a] * A[a, k] for a in self._actions for k in range(self._K))
        m.setObjective(obj, GRB.MAXIMIZE)

        m.optimize()

        # open-loop optimal action sequence
        open_loop = [None for _ in range(self._K)]
        for idx, v in A.items():
            if v.X > 0.5:
                a, k = idx
                open_loop[k] = a

        # print('open loop :', open_loop)
        info = {'runtime': m.Runtime}

        return open_loop, info
