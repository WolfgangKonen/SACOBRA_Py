import numpy as np


class EQUoptions:
    """
        Equality handling options
    """
    def __init__(self,
                 active=True,
                 equEpsFinal=1e-07,
                 initType="TAV",  # "useGrange", "TAV", "TMV", "EMV"
                 epsType="expFunc",
                 dec=1.5,
                 refine=True,
                 refineMaxit=100
                 ):
        self.active = active
        self.equEpsFinal = equEpsFinal
        self.initType = initType
        self.epsType = epsType
        self.dec = dec
        self.refine = refine
        self.refineMaxit = refineMaxit
