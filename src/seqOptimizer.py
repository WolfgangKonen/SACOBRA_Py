import numpy as np

# need to specify SACOBRA_Py.src as source folder in File - Settings - Project Structure,
# then the following import statements will work:
from cobraInit import CobraInitializer
from innerFuncs import verboseprint
from optimizer import NLoptOptimizer, DEoptOptimizer
from phase2Vars import Phase2Vars


class SeqOptimizer:
    """
    Sequential optimizer for phase II that optimizes on the surrogates in each sequential step.

    Several optimizer algorithms from `nlopt <https://nlopt.readthedocs.io/en/latest/>`_, the nonlinear optimization
    package **NLopt** and from ``scipy``'s differential evolution method are available and can be selected
    via parameter ``cobra.sac_opts.SEQ.optimizer``, namely

    - ``COBYLA``: `nlopt.LN_COBYLA <https://nlopt.readthedocs.io/en/latest/NLopt_Python_Reference/#the-nloptopt-class>`_
    - ``ISRESN``: `nlopt.GN_ISRES <https://nlopt.readthedocs.io/en/latest/NLopt_Python_Reference/#the-nloptopt-class>`_
    - ``DE_OPT``: `scipy.optimize.differential_evolution <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html>`_

    :param xStart: where (in input space) the optimizer starts
    :param cobra: parameter ``SEQ.optimizer`` in ``cobra.sac_opts`` specifies the optimization algorithm
    :param p2: object ``p2.opt_res`` of class :class:`.OptimResult` contains the optimization results
    """
    def __init__(self, xStart: np.ndarray, cobra: CobraInitializer, p2: Phase2Vars):
        """
        """
        s_opts = cobra.sac_opts
        verboseprint(s_opts.verbose, important=False,
                     message=f"[SeqOptimizer] {s_opts.SEQ.optimizer} optimization on surrogate ...")

        o = s_opts.SEQ.optimizer
        if o == "COBYLA":
            the_opt =  NLoptOptimizer(xStart, cobra, p2)
        elif o == "ISRESN":
            the_opt =  NLoptOptimizer(xStart, cobra, p2)
        elif o == "DE_OPT":
            the_opt =  DEoptOptimizer(xStart, cobra, p2)
        else:
            raise RuntimeError(f"Optimizer {s_opts.SEQ.optimizer} is not (yet) implemented")
        p2.opt_res = the_opt.get_result()

        # TODO: add other sequential optimizers

        verboseprint(s_opts.verbose, False,
                     f"[SeqOptimizer] {p2.ro}: {p2.opt_res.x} ... finished ({p2.opt_res.time_ms} msec)")

# TODO:
# if (cobra$seqOptimizer == ")ISRESCOBYLA") {
#     assert ("cobra$nCobyla is not set to a numeric",is.numeric(cobra$nCobyla));
#     assert ("cobra$nIsres is not set to a numeric",is.numeric(cobra$nIsres));


# # -----------------------------------------------------------------------------------------------
# # ----------------  helper functions     subprob*, gCOBRA*  -------------------------------------
# # -----------------------------------------------------------------------------------------------


#
# # --- TODO other subProb's for Phase I (currently commented out)
#

#     # surrogate evaluation of 'f' for constrained optimization methods  - phase I
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

# # surrogate evaluation of '\vec{g}' for constrained optimization methods  - phase I
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
