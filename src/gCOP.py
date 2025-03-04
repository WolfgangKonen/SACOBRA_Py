import numpy as np

class GCOP:
    """
    Constraint Optimization Problem Benchmark (G Function Suite)

    Instantiate problem G01 with ``G01 = GCOP("G01")``.

    Only problems G02 and G03 have extra parameter ``dimension``.
    All other problems G01, G04, ..., G24 have fixed dimensions.
    """
    def __init__(self, name, dimension=None):
        all_names = [f"G{i+1:02d}" for i in range(24)]
        assert name in all_names, f"{name} is not an allowed G-function name"
        self.name = name
        self.xStart = None
        self.info = None
        if name == "G01": self._call_G01()
        elif name == "G02": self._call_G02(dimension)
        elif name == "G05": self._call_G05()
        elif name == "G11": self._call_G11()
        else:
            str = f"G-function {name} not yet implemented"
            raise NotImplementedError(str)

    def _call_G01(self):
        self.dimension = 13
        self.lower = np.repeat(0, self.dimension)
        self.upper = np.concatenate((np.repeat(1,9), np.repeat(100, 3), 1), axis=None)
        self.nConstraints = 9
        self.is_equ = np.repeat(False, 9)
        self.solu = np.concatenate((np.repeat(1,9), np.repeat(3, 3), 1), axis=None)
        self.xStart = np.repeat(0, self.dimension)
        self.fn = lambda x: np.array([np.sum(5*x[0:4])-(5*sum(x[0:4]*x[0:4]))-(sum(x[4:13])),
                                      (2*x[0]+2*x[1]+x[9]+x[10] - 10),
                                      (2*x[0]+2*x[2]+x[9]+x[11] - 10),
                                      (2*x[1]+2*x[2]+x[10]+x[11] - 10),
                                      -8*x[0]+x[9],
                                      -8*x[1]+x[10],
                                      -8*x[2]+x[11],
                                      -2*x[3]-x[4]+x[ 9],
                                      -2*x[5]-x[6]+x[10],
                                      -2*x[7]-x[8]+x[11]])

    def _call_G02(self, dim):
        assert dim is not None, "dimension has to be integer, not None"
        self.dimension = dim
        self.lower = np.repeat(1e-16, dim)
        self.upper = np.repeat(19, dim)
        self.nConstraints = 2
        self.is_equ = np.repeat(False, 2)
        if dim == 2: self.solu = np.array([1.600859, 0.4684985])
        elif dim == 10: self.solu = np.array([3.1238477, 3.0690696, 3.0139085,
                                              2.9572856, 1.4654789, 0.3684877,
                                              0.3633289, 0.3592627, 0.3547453, 0.3510025] )
        else: self.solu = None
        # no xStart provided
        def denom(x):
            return np.sqrt(np.sum(np.array([(i+1)*x[i] for i in range(dim)])))
        self.fn = lambda x: np.array([-np.abs(np.sum(np.cos(x)**4)-(2*np.prod(np.cos(x)**2))/denom(x)),
                                      0.75-np.prod(x),
                                      (np.sum(x)-7.5*dim)])

    def _call_G05(self):
        self.dimension = 4
        self.lower = np.concatenate(np.repeat(0,2), np.repeat(-0.55, 2))
        self.upper = np.concatenate(np.repeat(1200, 2), np.repeat(0.55, 2))
        self.nConstraints = 5
        self.is_equ = np.array([False, False, True, True, True])
        self.solu = np.array([679.94531748791177961,
                              1026.06713513571594376,
                              0.11887636617838561,
                              -0.39623355240329272])
        self.xStart = np.concatenate(np.random.rand(2)*1200,            # x1, x2
                                     0.55*(np.random.rand(2)*2 - 1))     # x3, x4
        self.fn = lambda x: np.array([3*x[0]+1e-6*(x[0]**3)+2*x[1]+(2*1e-6/3)*(x[1]**3),
                                      x[2] - x[3] - 0.55,
                                      x[3] - x[2] - 0.55,
                                      1000 * np.sin(-x[2] - 0.25) + 1000 * np.sin(-x[3] - 0.25) + 894.8 - x[0],
                                      1000 * np.sin( x[2] - 0.25) + 1000 * np.sin( x[2] - x[3] - 0.25) + 894.8 - x[1],
                                      1000 * np.sin( x[3] - 0.25) + 1000 * np.sin( x[3] - x[2] - 0.25) + 1294.8])

    def _call_G11(self):
        self.dimension = 2
        self.lower = np.array([-1, -1])
        self.upper = np.array([1, 1])
        self.nConstraints = 1
        self.is_equ = np.array([True])
        self.solu = np.array([-np.sqrt(0.5), 0.5])
        self.fn = lambda x: np.array([x[0]**2 + (x[1]-1)**2,
                                      x[1] - x[0]**2 ])


if __name__ == '__main__':
    G01 = GCOP("G01")
    print(G01.solu)
    print(G01.fn(G01.solu))
    G02 = GCOP("G02", 10)
    print(G02.solu)
    print(G02.fn(G02.solu))
    G11 = GCOP("G11", 10)
    print(G11.solu)
    print(G11.fn(G11.solu))