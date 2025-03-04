import numpy as np


class TRoptions:
    """
        Trust region options
    """
    def __init__(self,
                 active=False,
                 shape="cube",
                 radiMin=0.01,
                 radiMax=0.8,
                 radiInit=0.1,
                 center="xbest"
                 ):
        self.active = active
        self.shape = shape
        self.radiMin = radiMin
        self.radiMax = radiMax
        self.radiInit = radiInit
        self.center = center
