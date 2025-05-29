import numpy as np


class SEQoptions:
    """
    Options for the sequential optimization

    :param optimizer: string defining the optimization method for SACOBRA_Py phase I and II. One out of ["COBYLA","ISRESN"]
    :param feval: maximum number of function evaluations on the surrogate model
    :param tol: convergence tolerance for sequential optimizer
    :param conTol: constraint violation tolerance
    :param penaF: (TODO)
    :param sigmaD: (TODO)
    :param epsilonInit: initial constant added to each constraint to maintain a certain margin to boundary
    :param epsilonMax: maximum for constant added to each constraint
    :param finalEpsXiZero: if True, set in final iteration ``EPS`` and ``XI`` to zero for best exploitation
    :param trueFuncForSurrogates: if True, use the true (constraint & fitness) functions instead of surrogates (only for debug analysis)
    """
    def __init__(self,
                 optimizer="COBYLA",
                 feval=1000,
                 tol=1e-6,
                 conTol=0.0,
                 penaF=[3.0, 1.7, 3e5],
                 sigmaD=[3.0, 2.0, 100],
                 epsilonInit=None, epsilonMax=None,
                 finalEpsXiZero=True,  # if True, then set EPS=XI=0 in final iteration (full exploit, might require
                                       # SEQ.conTol=1e-7 instead of 0.0)
                 trueFuncForSurrogates=False
                 ):
        self.optimizer = optimizer
        self.feval = feval
        self.tol = tol
        self.conTol = conTol
        self.penaF = penaF
        self.sigmaD = sigmaD
        self.epsilonInit = epsilonInit
        self.epsilonMax = epsilonMax
        self.finalEpsXiZero = finalEpsXiZero
        self.trueFuncForSurrogates = trueFuncForSurrogates
