class OptimResult:
    """
    Optimization results for :class:`.SeqOptimizer`

    :param x:       best solution found
    :param minf:    objective at best solution
    :param res_code: result message or code
    :param fe_max:  number of function evalutions
    :param time_ms: optimization time in msec
    """
    def __init__(self, x, minf, res_code, fe_max, time_ms):
        """
        """
        self.x = x
        self.minf = minf
        self.res_code=res_code
        self.fe_max = fe_max,
        self.time_ms = time_ms
