import time
import nlopt
import numpy as np
# need to specify SACOBRA_Py.src as source folder in File - Settings - Project Structure,
# then the following import statements will work:
from cobraInit import CobraInitializer
from innerFuncs import distLine, verboseprint
from phase2Vars import Phase2Vars


class SeqOptimizer:
    """
    Sequential optimizer for phase II that optimizes on the surrogates in each sequential step.

    Several optimizer algorithms from `nlopt <https://nlopt.readthedocs.io/en/latest/>`_, the nonlinear optimization
    package **NLopt**, are available and can be selected via parameter ``cobra.sac_opts.SEQ.optimizer``, namely

    - ``COBYLA``: `nlopt.LN_COBYLA <https://nlopt.readthedocs.io/en/latest/NLopt_Python_Reference/#the-nloptopt-class>`_
    - ``ISRESN``: `nlopt.GN_ISRES <https://nlopt.readthedocs.io/en/latest/NLopt_Python_Reference/#the-nloptopt-class>`_

    :param xStart: where (in input space) the optimizer starts
    :param cobra: parameter ``cobra.sac_opts.SEQ.optimizer`` specifies the optimization algorithm
    :param p2: dict ``p2.opt_res`` contains the optimization results
    """
    def __init__(self, xStart: np.ndarray, cobra: CobraInitializer, p2: Phase2Vars):
        """
        """
        s_opts = cobra.sac_opts
        s_res = cobra.sac_res
        start = time.perf_counter()
        verboseprint(s_opts.verbose, important=False,
                     message=f"[SeqOptimizer] {s_opts.SEQ.optimizer} optimization on surrogate ...")

        # construct **now** sf_factory with methods subProb2 and gCOBRA, given the current state of cobra and p2
        sf_factory = SeqFuncFactory(cobra, p2)

        # cat("Starting optimizer ...\n");
        switcher = {
            "COBYLA": nlopt.opt(nlopt.LN_COBYLA, xStart.size),
            "ISRESN": nlopt.opt(nlopt.GN_ISRES, xStart.size)
        }
        opt = switcher.get(s_opts.SEQ.optimizer, "not implemented")
        assert opt != "not implemented", f"Optimizer {s_opts.SEQ.optimizer} is not (yet) implemented"

        # TODO: add other sequential optimizers

        lower = s_res['lower']
        upper = s_res['upper']
        assert (xStart >= lower).all(), "[SeqOptimizer] xStart >= lower violated"
        assert (xStart <= upper).all(), "[SeqOptimizer] xStart <= upper violated"
        opt.set_lower_bounds(lower)
        opt.set_upper_bounds(upper)
        opt.set_min_objective(sf_factory.subProb2)
        tol_i = sf_factory.get_tol_i()
        if tol_i.size > 0:  # this should always be the case (due to dist req constraint)
            opt.add_inequality_mconstraint(sf_factory.g_vec_c, tol_i)
        tol_e = sf_factory.get_tol_e()
        if tol_e.size > 0:
            opt.add_equality_mconstraint(sf_factory.h_vec_c, tol_e)
        opt.set_xtol_rel(s_opts.SEQ.tol)
        opt.set_maxeval(s_opts.SEQ.feMax)

        try:
            x = opt.optimize(xStart)
        except nlopt.RoundoffLimited:
            print(f"WARNING: seqOpt [{p2.num}] nlopt.RoundoffLimited exception "
                  f"(result code {opt.last_optimize_result()})")
            x = xStart.copy()

        minf = opt.last_optimum_value()
        time_ms = (time.perf_counter() - start) * 1000
        p2.opt_res = {'x': x,
                      'minf': minf,
                      'res_code': opt.last_optimize_result(),
                      'feMax': opt.get_numevals(),
                      'time_ms': time_ms
                      }
        verboseprint(s_opts.verbose, False,
                     f"[SeqOptimizer] {p2.ro}: {x} ... finished ({time_ms} msec)")


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
        # DON'T change here to 'self.is_equ is False' as the PEP hint suggest --> strange error in NLopt (!)
        self.ine_ind = np.concatenate((0, self.ine_ind), axis=None)
        tol = 0.0           # here we use always 0.0, the constraint tolerance conTol is used in EvaluatorReal
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
        else:
            y = self.p2.fitnessSurrogate(x)[0]

        return y

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

        distance = distLine(x, self.cobra.sac_res['A'])
        subC = np.maximum((self.p2.ro - distance), 0)
        # h[1] = sum(subC)*cobra$drFactor
        h = sum(subC)

        DBG = False
        if DBG and h > 0:
            print(f"gCOBRA:  {h} {max(constraintPrediction)}")

        if nConstraints > 0:
            h = (-1.0) * np.concatenate((h, constraintPrediction), axis=None)
                # TODO -1* ... is required for COBYLA constraints, maybe also for other optimizers?
        else:
            h = -h
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


def calcConstrPred(x, cobra: CobraInitializer, p2: Phase2Vars) -> np.ndarray:
    """
    Calculate constraint_prediction at point ``x``. This includes

    - the proper handling of ``s_opts.EQU.active`` and
    - ``s_opts.SEQ.trueFuncForSurrogates``

    Known callers:

    - subprobPhaseI, subprob2PhaseI, gCOBRAPhaseI (all phase I), and
    - subprob, gCOBRA (phase II)

    :param x: the point to evaluate
    :param cobra: we need here members ``sac_opts`` and ``sac_res``
    :param p2: we need here members ``constraintSurrogates``, ``mu4`` and ``EPS``
    :return: a vector of length ``nConstraints``
    """
    s_opts = cobra.sac_opts
    s_res = cobra.sac_res
    if s_opts.EQU.active:
        # We form here the vector
        #
        #     ( g_i(x), h_j(x) - mu, - h_j(x) - mu ) + eps**2
        #
        # with mu = currentMu, eps=p2.EPS. This vector should be in all components <= 0
        # in order to fulfill the constraints
        #
        currentMu = s_res['muVec'][-1]
        currentMu = p2.mu4  # normally 0. Experimental: same value as currentMu, applied to *inequalities*
        if s_opts.SEQ.trueFuncForSurrogates:
            constraint_pred1 = s_res['fn'](x)[1:]
        else:
            constraint_pred1 = p2.constraintSurrogates(x)[0]    # why [0]? - constraintSurrogates returns a
            # (1,nC)-matrix, but we want a (nc,)-vector here (nC = nConstraints)

        ine_ind = np.flatnonzero(s_res['is_equ'] == False)
        equ_ind = np.flatnonzero(s_res['is_equ'])
        constraint_pred1[ine_ind] = constraint_pred1[ine_ind] - currentMu    # g(x) - mu, new 2025/04/02
        constraint_pred1[equ_ind] = constraint_pred1[equ_ind] - currentMu   # this creates h(x)-mu
        constraint_pred2 = -constraint_pred1[equ_ind] - 2 * currentMu       # this creates -h(x)-mu
        # why 2*currentMu? - because we modify the already created h(x)-mu to -(h(x)-mu)-2*mu = -h(x)-mu

        constraint_prediction = np.concatenate((constraint_pred1, constraint_pred2), axis=None) + p2.EPS ** 2

    else:  # i.e. if not s_opts.EQU.active
        if s_opts.SEQ.trueFuncForSurrogates:
            constraint_prediction = s_res['fn'](x)[1:] + p2.EPS ** 2
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
# TODO:
# if (cobra$seqOptimizer == ")ISRESCOBYLA") {
#     assert ("cobra$nCobyla is not set to a numeric",is.numeric(cobra$nCobyla));
#     assert ("cobra$nIsres is not set to a numeric",is.numeric(cobra$nIsres));


# # -----------------------------------------------------------------------------------------------
# # ----------------  helper functions     subprob*, gCOBRA*  -------------------------------------
# # -----------------------------------------------------------------------------------------------


def subProbPhaseI(x, cobra: CobraInitializer, p2: Phase2Vars):
    """
        surrogate penalty function for unconstrained optimization methods  - PHASE I

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

#
# # --- TODO other subProb's for Phase I (currently commented out)
#

#     # surrogate evaluation of 'f' for constrained optimization methods  - PHASE I
#     subProb2PhaseI < - function(x, cobra)
#     {
#
#     if (any( is.nan(x))){
#     warning("subProb2PhaseI: x value is NaN, returning Inf")
#     return (Inf)
#     }
#     # y<-predict.RBFinter(constraintSurrogates,matrix(x,ncol=dimension))
#
#     constraintPrediction < - calcConstrPred(x, cobra);
#
#     y < - sum(max(constraintPrediction, 0) ^ 2)
#     return (y)
#
# }

# # surrogate evaluation of '\vec{g}' for constrained optimization methods  - PHASE I
# gCOBRAPhaseI < - function(x, cobra)
# {
#
# h < - c()
# distance < - distLine(x, cobra$A)
# subC < -pmax((cobra$ro-distance), 0)
# h[1] < - sum(subC)
#
# constraintPrediction < - calcConstrPred(x, cobra);
#
# h < - (-1.0) * c(h[1], constraintPrediction)
#       # TODO -1* ... is required for COBYLA constraints, maybe also for other optimizers?
# return (h)
# }

# # surrogate penalty function for unconstrained optimization methods  - PHASE II
# subProb < -function(x, cobra)
# {
# # x<-x%*%cobra$TM
# # x<-as.numeric(x)
# # browser()
#
# if (any( is.nan(x))){
# warning("subProb: x value is NaN, returning Inf")
# return (Inf)
# }
# # cat(">>> Entering subProb\n")
# distance < - distLine(x, cobra$A)
# subC < -pmax((cobra$ro-distance), 0)
# penalty1 < -sum(subC)
# penalty2 < -0
# if (cobra$CONSTRAINED)
# {
# constraintPrediction < - calcConstrPred(x, cobra);
# # if (cobra$trueFuncForSurrogates) constraintPrediction <-  cobra$fn(x)[-1]+cobra$EPS^2
# maxViolation < - sapply(1: length(constraintPrediction), FUN = function(i)
# max(0, constraintPrediction[i]))
# penalty2 < - sum(maxViolation)
# }
#
# y < - subProbConstraintHandling(x, cobra, penalty1, penalty2, maxViolation, cobra$sigmaD, cobra$penaF)
# if (cobra$trueFuncForSurrogates) y < - cobra$fn(x)[1] + (penalty1 * cobra$sigmaD[1] * 100 + penalty2) * cobra$penaF[1]
# # cat(">>SubProb: ", interpRBF(x, cobra$fitnessSurrogate), " / ",
# #     (C * cobra$feMax)^alpha * sum(maxViolation)^beta , " ||",
# #     (C * cobra$feMax)^alpha, " / ", sum(maxViolation)^beta ,"\n")
# # cat("<<< leaving subProb\n")
# return (y)
# }  # subProb()
#

# # gCOBRA_cobyla <- function(x,cobra) {
# #   return(-gCOBRA(x,cobra))
# # }

# isresCobyla < - function(xStart, fn=subProb2, hin=gCOBRA, cobra)
# {
# # maxeval=cobra$seqFeval;
# # subMin$feMax=subMin$iter;
# hin_c = cobra$gCOBRA_c
# subMin1 < - isres2(xStart, fn=fn,
#                    lower=cobra$lower, upper = cobra$upper, hin = hin, maxeval = cobra$seqFeval, cobra = cobra);
# subMin2 < - nloptr::cobyla(xStart, fn=fn, lower=cobra$lower, upper = cobra$upper, hin = hin_c, control = list(
#     maxeval=cobra$seqFeval, xtol_rel = cobra$seqTol), cobra = cobra, deprecatedBehavior = FALSE);
# # subMin2 <- nloptr::cobyla(xStart,fn=fn,lower=cobra$lower,upper=cobra$upper,hin=hin,
# #                           control=list(maxeval=cobra$seqFeval,xtol_rel=cobra$seqTol), cobra=cobra);
# if (subMin1$value < subMin2$value) {
# subMin1$nIsres < - cobra$nIsres+1;
# subMin1$nCobyla < - cobra$nCobyla;
# return (subMin1);
# } else {
# subMin2$nCobyla < - cobra$nCobyla + 1;
# subMin1$nIsres < - cobra$nIsres;
# return (subMin2);
# }
# }
#

#
# # constraint handling for suProb, the surrogate penalty function for unconstrained
# # optimization methods in cobraPhaseII.R
# subProbConstraintHandling < - function(x, cobra, penalty1, penalty2, maxViolation, sigmaD, penaF)
# {
# switch(cobra$constraintHandling,
#
#     # J.A. Joines and C.R. Houck:  On the use of non-stationary penalty functions to solve
#     # nonlinear constrained optimization problems with GA's. In Proceedings of the First IEEE
#     # Conference on Evolutionary Computation, p. 579-584, (1994)
# JOINESHOUCK = {
#     C = 0.5
# alpha = 2
# beta = 2
# y < -interpRBF(x, cobra$fitnessSurrogate) + (C * cobra$feMax) ^ alpha * sum(maxViolation) ^ beta
# },
#
# # A.E. Smith and D.M. Tate: Genetic optimization using a penalty function. In Proceedings of the
# # Fifth International Conference on Genetic Algorithms p. 499-505, (1993)
# SMITHTATE = {
#     # TODO
#     lambda = 1
#            beta1 = 3
#            beta2 = 2
#            kappa = 1
#            d = sum(maxViolation) ^ kappa
#            y < - interpRBF(x, cobra$fitnessSurrogate) + penalty
# },
#
# # D.W. Coit, A.E. Smith and D.M. Tate: Adaptive penalty methods for genetic optimization of
# # constrained combinatorial problems. In INFORMS Journal on Computing 8(2), (1996)
# COIT = {
#     fFeas < - cobra$fbest
# fAll < - min(cobra$Fres)
# NFT < - rep(0.05, cobra$nConstraints)
# kappa < - 1
# y < - interpRBF(x, cobra$fitnessSurrogate) + (fFeas - fAll) * sum((maxViolation / NFT) ^ kappa)
# },
#
# # T. Baeck and S. Khuri: An evolutionary heuristic for the maximum independent set problem
# # In Proceedings of the First IEEE Conference on Evolutionary Computation, p. 531-535, (1994)
# BAECKKHURI = {
#     # TODO
#     K = 10 ^ 9
# p = cobra$nConstraints
# s < - length(cobra$numViol)
# penalty < - K - s * (K / cobra$nConstraints)
# y < - interpRBF(x, cobra$fitnessSurrogate) + penalty
# },
#
# DEFAULT = {
#     y < -interpRBF(x, cobra$fitnessSurrogate) + (penalty1 * sigmaD[1] + penalty2) * penaF[1]
# }
# )
#
# # PKDEBUG=TRUE|FALSE
# # if(PKDEBUG){# Joines and Houck 1994
# #  C=5
# #  alpha=2
# #  beta=2
# #  y<-interpRBF(x, cobra$fitnessSurrogate) + (C * cobra$feMax)^alpha * sum(maxViolation)^beta
# # }else{y<-interpRBF(x, cobra$fitnessSurrogate) + (penalty1*sigmaD[1] + penalty2)*penaF[1]}
#
# return (y)
# }
