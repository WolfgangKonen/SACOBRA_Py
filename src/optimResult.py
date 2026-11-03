import numpy as np


class OptimResult:
    """
    Optimization results for :class:`.SeqOptimizer`

    :param optimizer:       the name of the optimizer
    :param x:       best solution found
    :param minf:    objective at best solution
    :param res_code: result or code
    :param res_msg: result message
    :param fe_max:  number of function evaluations
    :param time_ms: optimization time in msec
    """
    def __init__(self, optimizer: str, x: np.ndarray, minf: float,
                 res_code: int, res_msg: str, fe_max: int, time_ms: float):
        self.optimizer = optimizer
        self.x = x
        self.minf = minf
        self.res_code=res_code
        self.res_msg = "" if res_msg is None else res_msg
        self.fe_max = fe_max,
        self.time_ms = time_ms
