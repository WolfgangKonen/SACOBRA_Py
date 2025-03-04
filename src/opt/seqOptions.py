import numpy as np


class SEQoptions:
    def __init__(self,
                 optimizer="COBYLA",
                 feval=1000,
                 tol=1e-6,
                 conTol=0.0,
                 trueFuncForSurrogates=False
                 ):
        self.optimizer = optimizer
        self.feval = feval
        self.tol = tol
        self.conTol = conTol
        self.trueFuncForSurrogates = trueFuncForSurrogates
