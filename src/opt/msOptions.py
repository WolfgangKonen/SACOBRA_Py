import numpy as np


class MSoptions:
    """
         Model selection options
    """
    def __init__(self,
                 active=False,
                 apply=True,
                 models=["cubic", "MQ"],
                 widths=[0.01, 0.1, 1, 10],
                 freq=1,
                 slidingW=False,
                 winS=10,
                 quant=3,  # 3: median, 2: 25%, 4:75%
                 considerXI=False
                 ):
        """

        :param active:
        :param apply:
        :param models:
        :param widths:
        :param freq:
        :param slidingW:
        :param winS:
        :param quant:
        :param considerXI:
        """
        self.active = active
        self.apply = apply
        self.models = models
        self.widths = widths
        self.freq = freq
        self.slidingW = slidingW
        self.winS = winS
        self.quant = quant
        self.considerXI = considerXI
