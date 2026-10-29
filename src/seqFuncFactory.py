import numpy as np
from scipy.optimize import NonlinearConstraint

from cobraInit import CobraInitializer
from innerFuncs import verboseprint, distLine
from phase2Vars import Phase2Vars


class SeqFuncFactory:
    """
    Factory for the functions that :class:`SeqOptimizer` optimizes in phase II
    """
    def __init__(self, cobra: CobraInitializer, p2: Phase2Vars):
        self.cobra = cobra
        self.p2 = p2
        self.is_equ = cobra.sac_res['is_equ']
        # equ_ind is an index to the equality constraints in gCOBRA ('+ 1' because the first constraint is dist req):
        self.equ_ind = np.flatnonzero(self.is_equ) + 1
        # ine_ind is an index to the inequality constraints in gCOBRA
        # ['+ 1' and 'concatenate((0,...))' because the first inequality constraint is dist req]:
        self.ine_ind = np.flatnonzero(self.is_equ == False) + 1
        # DON'T change here to 'self.is_equ is False' as the PEP hint suggests --> strange error in NLopt (!)
        self.ine_ind = np.concatenate((0, self.ine_ind), axis=None)
        tol = 0.0           # here we use always 0.0, the constraint tolerance conTol is used in EvaluatorReal
        tol = self.cobra.sac_opts.SEQ.conTol
        self.tol_e = np.repeat(tol, self.equ_ind.size)      # we need tol_e and tol_i
        self.tol_i = np.repeat(tol, self.ine_ind.size)      # just to transport the sizes

    def subProb2(self, x, grad):
        """
        surrogate evaluation of 'f' for constrained optimization methods  - PHASE II
        """
        if np.isnan(x).any():
            verboseprint(self.cobra.sac_opts.verbose, True,
                         "[subProb2]: x value is NaN, returning Inf")
            return np.inf

        if self.cobra.sac_opts.SEQ.trueFuncForSurrogates:
            y = self.cobra.sac_res['fn'](x)[0]
            self.cobra.sac_res['ncall'][7] += 1  # ncall-debug
        else:
            y = self.p2.fitnessSurrogate(x)[0]

        return y

    def subProb2_de(self, x):
        return self.subProb2(x, 0.0)

    def gCOBRA(self, x, grad) -> np.array:
        """
        surrogate evaluation of '\vec{g}' for constrained optimization methods - PHASE II
        """
        nConstraints = self.cobra.sac_res['nConstraints']
        if np.isnan(x).any():
            print("[gCOBRA] WARNING: x contains NaNs, returning Inf")
            return np.repeat(np.inf, nConstraints+1)

        if nConstraints > 0:
            constraintPrediction = calcConstrPred(x, self.cobra, self.p2)

        if self.cobra.sac_opts.SEQ.DRC_active:
            distance = distLine(x, self.cobra.sac_res['A'])
            subC = np.maximum((self.p2.ro - distance), 0)
            # h[1] = sum(subC)*cobra$drFactor
            h = sum(subC)
        else:
            h = 0.0

        DBG = False
        if DBG and h > 0:
            print(f"gCOBRA:  {h} {max(constraintPrediction)}")

        if nConstraints > 0:
            h = (-1.0) * np.concatenate((h, constraintPrediction), axis=None)
                # TODO -1* ... is required for COBYLA constraints, maybe also for other optimizers?
        else:
            h = -np.array([h])          # bug fix 2025-09-30
        return h

    def g_vec_c(self, result, x, grad):
        """ Vector-valued inequality constraints for nlopt

            Note the special signature with ``result`` which has to be a vector of size self.ine_ind.size
        """
        g = -self.gCOBRA(x, grad)  # new convention g_i <= 0.
        result[:] = g[self.ine_ind]

    def h_vec_c(self, result, x, grad):
        """ Vector-valued equality constraints for nlopt

            Note the special signature with ``result`` which has to be a vector of size self.equ_ind.size
        """
        h = -self.gCOBRA(x, grad)  # new convention h_i <= 0.
        result[:] = h[self.equ_ind]

    def get_tol_e(self):
        """ Tolerance vector for equality constraints, length = # eq. constr.
        :return: tolerance vector
        """
        return self.tol_e

    def get_tol_i(self):
        """ Tolerance vector for inequality constraints, length = # problem ineq. constr. + 1 (dist requirement)
        :return: tolerance vector
        """
        return self.tol_i

    # def gCOBRA_c(self,x):
    #     # inequality constraints for nloptr::cobyla
    #     return -self.gCOBRA(x, None)  # new convention h_i <= 0.

    def check_if_solu_better(self, x, minf):
        if self.cobra.solu is None:
            return x, minf

        s = self.cobra.solu
        fs = self.cobra.rw.forward(s)
        y = self.p2.fitnessSurrogate(fs)[0]
        nConstraints = self.cobra.sac_res['nConstraints']
        conTol = self.cobra.sac_opts.SEQ.conTol
        if nConstraints > 0:
            constraintPrediction = calcConstrPred(fs, self.cobra, self.p2)
        s_feasible = np.all(constraintPrediction < conTol)
        # print(minf, y, constraintPrediction)
        print(self.p2.num, minf, y, s_feasible)
        dummy = 0
        if y < minf and s_feasible:
            minf = y
            x[:] = fs[:]
        return x, minf


class ConstrFactory:
    """
    Factory for the constraints that :class:`DEoptOptimizer` needs
    """
    def __init__(self, sf: SeqFuncFactory):
        self.sf = sf
        self.cur_x = sf.cobra.sac_res['x0'] * 0.0
        self.conTol = sf.cobra.sac_opts.SEQ.conTol
        self.currentMu = sf.cobra.sac_res['muVec'][-1]

    def g_constr(self, i):
        g_func = lambda x: -self.gCOBRA_de(x)[self.sf.ine_ind[i]]
        # if i == 0: g_func = lambda x: -2.0     # only debug
        return NonlinearConstraint(g_func, -np.inf, self.conTol)


    def h_constr(self, j):
        h_func = lambda x: -self.gCOBRA_de(x)[self.sf.equ_ind[j]]
        return NonlinearConstraint(h_func,  -self.currentMu-self.conTol, +self.currentMu+self.conTol)

    def gCOBRA_de(self, x):
        """
        surrogate evaluation of '\vec{g}' for constrained optimization methods - PHASE II
        """
        nConstraints = self.sf.cobra.sac_res['nConstraints']
        if np.isnan(x).any():
            print("[gCOBRA] WARNING: x contains NaNs, returning Inf")
            return np.repeat(np.inf, nConstraints+1)

        if nConstraints > 0:
            constraintPrediction = calcConstrPred_de(x, self.sf.cobra, self.sf.p2)

        if self.sf.cobra.sac_opts.SEQ.DRC_active:
            distance = distLine(x, self.sf.cobra.sac_res['A'])
            subC = np.maximum((self.sf.p2.ro - distance), 0)
            # h[1] = sum(subC)*cobra$drFactor
            h = sum(subC)
        else:
            h = 0.0

        DBG = False
        if DBG and h > 0:
            print(f"gCOBRA:  {h} {max(constraintPrediction)}")

        if nConstraints > 0:
            h = (-1.0) * np.concatenate((h, constraintPrediction), axis=None)
                # TODO -1* ... is required for COBYLA constraints, maybe also for other optimizers?
        else:
            h = -np.array([h])          # bug fix 2025-09-30
        return h



def calcConstrPred(x, cobra: CobraInitializer, p2: Phase2Vars) -> np.ndarray:
    """
    Calculate constraint prediction at point ``x``. This includes

    - the proper handling of ``s_opts.EQU.active`` and
    - ``s_opts.SEQ.trueFuncForSurrogates``

    Details: If there are ``m`` inequality constraints and ``r`` equality constraints, then we form the vector

        ( g_i(x) + eps**2, h_j(x) - mu, - h_j(x) - mu )

    with ``m+2*r`` components and with ``mu=currentMu``, ``eps=p2.EPS``. This vector should be in all components
    ``<= SEQ.contol`` in order to fulfill the constraints.

    Known callers:

    - subprobPhaseI, subprob2PhaseI, gCOBRAPhaseI (all phase I), and
    - subprob, gCOBRA (phase II)

    :param x: the point to evaluate
    :param cobra: we need here members ``sac_opts`` and ``sac_res``
    :param p2: we need here members ``constraintSurrogates``, ``mu4`` and ``EPS``
    :return: a vector of length ``m+2*r``
    """
    s_opts = cobra.sac_opts
    s_res = cobra.sac_res
    if s_opts.EQU.active:
        currentMu = s_res['muVec'][-1]
        # --- The following line with p2.mu4 was active until 2026-01-20, but I do not know why. It leads always to
        # --- currentMu = 0. We now return to the earlier behavior: currentMu = s_res['muVec'][-1]
        # currentMu = p2.mu4  # normally 0. Experimental: same value as currentMu, applied to *inequalities*
        # --- The right way to use mu also for inequalities
        if s_opts.SEQ.trueFuncForSurrogates:
            constraint_pred1 = s_res['fn'](x)[1:]
            s_res['ncall'][8] += 1  # ncall-debug
        else:
            constraint_pred1 = p2.constraintSurrogates(x)[0]    # why [0]? - constraintSurrogates returns a
            # (1,nC)-matrix, but we want a (nc,)-vector here (nC = nConstraints)

        ine_ind = np.flatnonzero(s_res['is_equ'] == False)
        equ_ind = np.flatnonzero(s_res['is_equ'])
        version = 2   # 0 | 1 | 2
        if version == 0:    # Version BEFORE 2025/09/14, wrong
            constraint_pred1[ine_ind] = constraint_pred1[ine_ind] - currentMu    # g(x) - mu, new 2025/04/02
            constraint_pred1[equ_ind] = constraint_pred1[equ_ind] - currentMu   # this creates h(x)-mu
            constraint_pred2 = -constraint_pred1[equ_ind] - 2 * currentMu       # this creates -h(x)-mu
            # why 2*currentMu? - because we modify the already created h(x)-mu to -(h(x)-mu)-2*mu = -h(x)-mu
            constraint_prediction = np.concatenate((constraint_pred1, constraint_pred2), axis=None) + p2.EPS ** 2
        elif version == 1:  # Version 2025/09/14 - 2026/01/20, partly wrong
            # --- BUG FIX 2025/09/14: + p2.EPS**2 only for inequality constraints (and NO '- currentMu'): ---
            constraint_pred1[ine_ind] = constraint_pred1[ine_ind] + p2.EPS ** 2    # g(x) + EPS ** 2 (!!)
            constraint_pred1[equ_ind] = constraint_pred1[equ_ind] - currentMu   # this creates h(x)-mu
            constraint_pred2 = -constraint_pred1[equ_ind] - 2 * currentMu       # this creates -h(x)-mu
            # why 2*currentMu? - because we modify the already created h(x)-mu to -(h(x)-mu)-2*mu = -h(x)-mu
            constraint_prediction = np.concatenate((constraint_pred1, constraint_pred2), axis=None)
        else:   # i. e. version == 2, after 2026/01/20
            # --- BUG FIX 2026/01/20: DON'T set currentMu=p2.mu4, but use p2.mu4 in inequalities ---
            constraint_pred1[ine_ind] = constraint_pred1[ine_ind] - p2.mu4 + p2.EPS**2  # g(x)-mu4 + EPS**2
            constraint_pred1[equ_ind] = constraint_pred1[equ_ind] - currentMu   # this creates h(x)-mu
            constraint_pred2 = -constraint_pred1[equ_ind] - 2 * currentMu       # this creates -h(x)-mu
            constraint_prediction = np.concatenate((constraint_pred1, constraint_pred2), axis=None)

    else:  # i.e. if not s_opts.EQU.active
        if s_opts.SEQ.trueFuncForSurrogates:
            constraint_prediction = s_res['fn'](x)[1:] + p2.EPS ** 2
            s_res['ncall'][9] += 1  # ncall-debug
        else:
            constraint_prediction = p2.constraintSurrogates(x) + p2.EPS ** 2

    return constraint_prediction


def calcConstrPred_de(x, cobra: CobraInitializer, p2: Phase2Vars) -> np.ndarray:
    """
    Calculate constraint prediction at point ``x`` for DE (differential evolution) optimizer. This includes

    - the proper handling of ``s_opts.EQU.active`` and
    - ``s_opts.SEQ.trueFuncForSurrogates``

    Details: If there are ``m`` inequality constraints and ``r`` equality constraints, then we form the vector

        ( g_i(x) + eps**2, h_j(x) )

    with ``m+r`` components and with ``eps=p2.EPS``. This vector should be in the first ``m`` components
    ``<= SEQ.contol`` and in the last ``r`` components in range ``-mu <= C <= +mu`` in order to fulfill the constraints.

    :param x: the point to evaluate
    :param cobra: we need here members ``sac_opts`` and ``sac_res``
    :param p2: we need here members ``constraintSurrogates``, ``mu4`` and ``EPS``
    :return: a vector of length ``m+r``
    """
    s_opts = cobra.sac_opts
    s_res = cobra.sac_res
    if s_opts.EQU.active:
        currentMu = s_res['muVec'][-1]
        # --- The following line with p2.mu4 was active until 2026-01-20, but I do not know why. It leads always to
        # --- currentMu = 0. We now return to the earlier behavior: currentMu = s_res['muVec'][-1]
        # currentMu = p2.mu4  # normally 0. Experimental: same value as currentMu, applied to *inequalities*
        # --- The right way to use mu also for inequalities
        if s_opts.SEQ.trueFuncForSurrogates:
            constraint_prediction = s_res['fn'](x)[1:]
            s_res['ncall'][8] += 1  # ncall-debug
        else:
            constraint_prediction = p2.constraintSurrogates(x)[0]    # why [0]? - constraintSurrogates returns a
            # (1,nC)-matrix, but we want a (nc,)-vector here (nC = nConstraints)

        ine_ind = np.flatnonzero(s_res['is_equ'] == False)
        equ_ind = np.flatnonzero(s_res['is_equ'])
        version = 2   # 0 | 1 | 2
        if version == 2:
            constraint_prediction[ine_ind] = constraint_prediction[ine_ind] + p2.EPS**2  # g(x) + EPS**2

    else:  # i.e. if not s_opts.EQU.active
        if s_opts.SEQ.trueFuncForSurrogates:
            constraint_prediction = s_res['fn'](x)[1:] + p2.EPS ** 2
            s_res['ncall'][9] += 1  # ncall-debug
        else:
            constraint_prediction = p2.constraintSurrogates(x) + p2.EPS ** 2

    return constraint_prediction


def check_if_cobra_optimizable(cobra: CobraInitializer, p2: Phase2Vars):
    s_res = cobra.sac_res
    assert type(p2.ro) is float or type(p2.ro) is np.float64, "p2.ro is not set to a numeric"
    assert type(p2.EPS) is float or type(p2.EPS) is np.float64, "p2.EPS is not set to a numeric"
    assert type(s_res['dimension']) is int, "cobra.sac_res['dimension'] is not set to an int"
    assert type(s_res['nConstraints']) is int, "cobra.sac_res['nConstraints'] is not set to an int"
    assert s_res['A'].ndim == 2, "cobra.sac_res['A'] is not a matrix"
    assert s_res['xStart'].ndim == 1, "cobra.sac_res['xStart'] is not a vector"
    assert type(s_res['fn']).__name__ in ['function', 'method'], "cobra.sac_res['fn'] is not a function"
    assert p2.fitnessSurrogate.__class__.__name__ == "RBFmodel", "p2.fitnessSurrogate is not RBFmodel"
    if s_res['nConstraints'] > 0:
        assert p2.constraintSurrogates.__class__.__name__ == "RBFmodel", "p2.constraintSurrogates is not RBFmodel"


def subProbPhaseI(x, cobra: CobraInitializer, p2: Phase2Vars):
    """
        surrogate penalty function for unconstrained optimization methods  - phase I

    :param x:       the input point
    :param cobra:   created by CobraInitializer
    :param p2:      phase II variables
    :return:
    """
    if np.isnan(x).any():
        print("[subProbPhaseI] WARNING: x contains NaNs, returning Inf")
        return np.inf

    distance = distLine(x, cobra.sac_res['A'])
    subC = np.maximum(cobra.sac_res['ro']-distance, 0)
    penalty1 = sum(subC)
    num = cobra.sac_res['A'].shape[0]

    constraintPrediction = calcConstrPred(x, cobra, p2)

    maxViolation = np.maximum(constraintPrediction, 0)
    # penalty2 <- sum(maxViolation)
    # Constraint handling: Coit et al. (1996)
    NFT = np.repeat(0.1, maxViolation.size)
    kappa = 2

    penalty2 = (maxViolation.size * np.sum(maxViolation) * num) * np.sum((maxViolation/NFT)**kappa)
    # y = interpRBF(x, fitnessSurrogate) + (penalty1*cobra$sigmaD[1] + penalty2)*cobra$penaF[1]
    # y = sum(max(interpRBF(x,constraintSurrogates),0)^2) + (penalty1*cobra$sigmaD[1] + penalty2)*cobra$penaF[1]

    # f = interpRBF(x, cobra$fitnessSurrogate)      # /WK/2016-05/ Bug fix: f is NOT part of subprobPhaseI
    # y = f  + penalty2 #+ penalty1*cobra$sigmaD[1] #
    y = penalty2  # + penalty1*cobra$sigmaD[1]
    return y
