import numpy as np

class RescaleWrapper:
    def __init__(self, fn, lower, upper, lb, ub):
        self.origfn = fn
        self.lower = lower
        self.upper = upper
        self.lb = lb
        self.ub = ub
        self.dim = self.lb.size

    def __call__(self, x):
        return self.apply(x)

    def apply(self, x):
        orig_x = np.array([np.interp(x[i], (self.lb[i], self.ub[i]), (self.lower[i], self.upper[i]))
                           for i in range(self.dim)])
        return self.origfn(orig_x)

    def forward(self, x):
        new_x = np.array([np.interp(x[i], (self.lower[i], self.upper[i]), (self.lb[i], self.ub[i]))
                           for i in range(self.dim)])
        return new_x

    def inverse(self, x):
        orig_x = np.array([np.interp(x[i], (self.lb[i], self.ub[i]), (self.lower[i], self.upper[i]))
                           for i in range(self.dim)])
        return orig_x

