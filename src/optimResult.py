class OptimResult:
    def __init__(self, x, minf, res_code, fe_max, time_ms):
        self.x = x
        self.minf = minf
        self.res_code=res_code
        self.fe_max = fe_max,
        self.time_ms = time_ms
