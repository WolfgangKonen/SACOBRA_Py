import numpy as np


class EQUoptions:
    """
        Equality handling options

        :param active:  if set to TRUE, the equality-handling (EH) technique is activated. The EH
          technique transforms each equality constraint :math:`h(\mathbf{x})=0` into two inequality
          constraints :math:`h(\mathbf{x})-\mu <0` and :math:`-h(\mathbf{x})-\mu<0` with an adaptive
          (normally decaying) margin :math:`\mu`.
        :param initType: the equality margin :math:`\mu` is initialized with one of these choices:
          ``["TAV" | "TMV" | "EMV" | "useGrange"]``
        :param muType: type of function used to shrink margin :math:`\mu` during the optimization process.
          One out of ``["SAexpFunc" | "expFunc" | "funcDim" | "funcSDim" | "Zhang" | "CONS"]``, see ``modifyMu``
          in ``equHandling.py``
        :param muDec: decay factor for margin :math:`\mu`, see ``modifyMu``
        :param muFinal: lower bound for margin :math:`\mu`. ``muFinal`` should be set to a small but non-zero value (larger than machine accuracy).
        :param muGrow: every ``muGrow`` (e.g. 100) iterations, re-enlarge the :math:`\mu`-band. If 0, then re-enlarge never
        :param mu4inequality: use the artificial feasibility band also for inequalities (experimental)
        :param refine: enables the :ref:`refine step <refineStep-label>` for equality handling
        :param refineMaxit: maximum number of iterations used in the :ref:`refine step <refineStep-label>`. Note that the refine
          step runs on the surrogate models and does not impose any extra real function evaluations
        :param refineAlgo: optimizer for :ref:`refine step <refineStep-label>` ["BFGS_0" | "BFGS_1" | "COBYQA" | "COBYLA"]
        :param refinePrint: whether to print "cg-values (before,after,true)" after each :ref:`refine step <refineStep-label>`
    """
    def __init__(self,
                 active=True,
                 initType="TAV",        # "useGrange" | "TAV" | "TMV" | "EMV"
                 muType="expFunc",      # expFunc | SAexpFunc | funcDim | funcSDim | Zhang | CONS
                 muDec=1.5,             # decay factor for mu
                 muFinal=1e-07,
                 mu4inequality=False,
                 muGrow=0,
                 refine=True,
                 refineMaxit=1000,      # 100,
                 refineAlgo="BFGS_0",   # BFGS_0 | BFGS_1 | COBYQA | COBYLA
                 refinePrint=False
                 ):
        self.active = active
        self.initType = initType
        self.muType = muType
        self.muDec = muDec
        self.muFinal = muFinal
        self.muGrow = muGrow
        self.mu4inequality = mu4inequality
        self.refine = refine
        self.refineMaxit = refineMaxit
        self.refineAlgo = refineAlgo
        self.refinePrint = refinePrint
