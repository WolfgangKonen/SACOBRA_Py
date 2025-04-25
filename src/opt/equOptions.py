import numpy as np


class EQUoptions:
    """
        Equality handling options
    """
    def __init__(self,
                 active=True,
                 equEpsFinal=1e-07,
                 initType="TAV",    # "useGrange", "TAV", "TMV", "EMV"
                 epsType="expFunc", # expFunc SAexpFunc funcDim funcSDim Zhang CONS
                 dec=1.5,
                 refine=True,
                 refineMaxit=1000, # 100,
                 muGrow=0,
                 mu4inequality=False
                 ):
        """

        :param active:  if set to TRUE, the equality-handling (EH) technique is activated. The EH
          technique transforms each equality constraint :math:`h(\mathbf{x})=0` into two inequality
          constraints :math:`h(\mathbf{x})-\mu <0` and :math:`-h(\mathbf{x})-\mu<0` with an adaptive
          decaying margin :math:`\mu`.
        :param equEpsFinal: lower bound for margin :math:`\mu`. ``equEpsFinal`` should be set
          to a small but non-zero value (larger than machine accuracy).
        :param initType: the equality margin :math:`\mu` is initialized with one of these choices:
          ``["TAV"|"TMV"|"EMV"|"useGrange"]``
        :param epsType: type of function used to shrink margin :math:`\mu` during the optimization process.
          One out of ``["SAexpFunc"|"expFunc"|"funcDim"|"funcSDim"|"Zhang"|"CONS"]``, see ``modifyMu``
        :param dec: decay factor for margin :math:`\mu`, see ``modifyMu``
        :param refine: enables the refine mechanism for equality handling
        :param refineMaxit: maximum number of iterations used in the refine step. Note that the refine
          step runs on the surrogate models and does not impose any extra real function evaluation
        :param muGrow: every muGrow (e.g. 100) iterations, re-enlarge the :math:`\mu`-band. If 0, then
          re-enlarge never
        """
        self.active = active
        self.equEpsFinal = equEpsFinal
        self.initType = initType
        self.epsType = epsType
        self.dec = dec
        self.refine = refine
        self.refineMaxit = refineMaxit
        self.muGrow = muGrow
        self.mu4inequality = mu4inequality
