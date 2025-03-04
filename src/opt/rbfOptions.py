import numpy as np


class RBFoptions:
    def __init__(self,
                 model="cubic",
                 width=-1,
                 widthFactor=1.0,
                 gaussRule="One",
                 rho=0.0,
                 degree= None
                 # ptail=True,
                 # squares=True
                 ):
        self.model = model
        self.width = width
        self.rho = rho
        self.widthFactor = widthFactor
        self.gaussRule = gaussRule
        self.degree = degree
        #self.ptail = ptail
        #self.squares = squares
