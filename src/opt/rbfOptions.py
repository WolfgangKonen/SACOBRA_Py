import numpy as np


class RBFoptions:
    def __init__(self,
                 model="cubic",
                 width=-1,
                 widthFactor=1.0,
                 gaussRule="One",
                 rho=0.0,
                 rhoDec=2.0,        # exponential decay factor for rho
                 rhoGrow=0,
                 degree= None
                 ):
        self.model = model
        self.width = width
        self.rho = rho
        self.rhoDec = rhoDec
        self.rhoGrow = rhoGrow
        self.widthFactor = widthFactor
        self.gaussRule = gaussRule
        self.degree = degree
