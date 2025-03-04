import numpy as np


class IDoptions:
    """
         Init design options
    """
    def __init__(self,
                 initDesign="RANDOM",  # other options: "RAND_R", "LHS", ... (see initDesigner)
                 initDesPoints: int = None,  # if None, cobraInit will set it to 2 * dimension + 1
                 initDesOptP: int = None,  # if None, cobraInit will set it to initDesPoints
                 initBias=0.005,
                 rescale=True,
                 newLower=-1,
                 newUpper=1
                 ):
        """
        :param initDesign:    options: "RANDOM", "RAND_R", "LHS", ... (see initDesigner)
        :param initDesPoints: number of initial design points. If None, cobraInit will set it to 2 * dimension + 1
        :param initDesOptP:
        :param initBias:
        :param rescale:
        :param newLower:
        :param newUpper:
        """
        self.initDesign = initDesign
        self.initDesPoints = initDesPoints
        self.initDesOptP = initDesOptP
        self.initBias = initBias
        self.rescale = rescale
        self.newLower = newLower
        self.newUpper = newUpper
