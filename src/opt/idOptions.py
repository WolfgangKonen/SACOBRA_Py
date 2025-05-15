

class IDoptions:
    """
    Init design options.

    See ``__init__(self, ...)`` for details.
    """
    def __init__(self,
                 initDesign="RANDOM",
                 initDesPoints: int = None,
                 initDesOptP: int = None,
                 initBias=0.005,
                 rescale=True,
                 newLower=-1,
                 newUpper=1
                 ):
        """
        Set the initial design options (d = input dimension of problem).

        :param initDesign:    options: "RANDOM", "RAND_R", "RAND_REP", "LHS", ... (see initDesigner)
        :param initDesPoints: number of initial design points. If None, cobraInit will set it to d+1 if RBF.degree=1
                              or to (d+1)(d+2)/2 if RBF.degree=2
        :param initDesOptP:   if None, cobraInit will set it to initDesPoints
        :param initBias:
        :param rescale:       if True, rescale input space from [lower, upper] to :math:`[newLower, newUpper]^d`
        :param newLower:      common new lower bound for each of the d input dimensions
        :param newUpper:      common new upper bound for each of the d input dimensions
        """
        self.initDesign = initDesign
        self.initDesPoints = initDesPoints
        self.initDesOptP = initDesOptP
        self.initBias = initBias
        self.rescale = rescale
        self.newLower = newLower
        self.newUpper = newUpper
