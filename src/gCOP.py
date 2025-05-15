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
        self.x0 = None
        self.info = None
        if name == "G01": self._call_G01()
        elif name == "G02": self._call_G02(dimension)
        elif name == "G03": self._call_G03(dimension)
        elif name == "G04": self._call_G04()
        elif name == "G05": self._call_G05()
        elif name == "G06": self._call_G06()
        elif name == "G07": self._call_G07()
        elif name == "G11": self._call_G11()
        elif name == "G13": self._call_G13()
        else:
            str = f"G-function {name} not yet implemented"
            raise NotImplementedError(str)

        if self.solu is None:
            self.fbest = None
        else:
            first_solu = self.solu[0, :] if self.solu.ndim == 2 else self.solu
            self.fbest = self.fn(first_solu)[0]     # objective at solution point

    def _call_G01(self):
        self.dimension = 13
        self.lower = np.repeat(0, self.dimension)
        self.upper = np.concatenate((np.repeat(1,9), np.repeat(100, 3), 1), axis=None)
        self.nConstraints = 9
        self.is_equ = np.repeat(False, 9)
        self.solu = np.concatenate((np.repeat(1,9), np.repeat(3, 3), 1), axis=None)
        self.x0 = np.repeat(0, self.dimension)
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
        assert dim is not None, "[call_G02] dimension has to be integer, not None"
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
        # no x0 provided

        def denom(x):
            return np.sqrt(np.sum(np.array([(i+1)*x[i] for i in range(dim)])))
        self.fn = lambda x: np.array([-np.abs(np.sum(np.cos(x)**4)-(2*np.prod(np.cos(x)**2))/denom(x)),
                                      0.75-np.prod(x),
                                      (np.sum(x)-7.5*dim)])

    def _call_G03(self, dim):
        assert dim is not None, "[call_G03] dimension has to be integer, not None"
        self.dimension = dim
        self.lower = np.repeat(0, dim)
        self.upper = np.repeat(1, dim)
        self.nConstraints = 1
        self.is_equ = np.repeat(True, 1)
        self.solu = np.repeat(1/np.sqrt(dim), dim)
        # no x0 provided
        self.fn = lambda x: np.array([-((np.sqrt(dim)) ** dim) * np.prod(x),
                                      np.sum(x*x)-1])

    def _call_G04(self):
        self.dimension = 5
        self.lower = np.concatenate(((78, 33), np.repeat(27, 3)))
        self.upper = np.concatenate(((102, 45), np.repeat(45, 3)))
        self.nConstraints = 6
        self.is_equ = np.repeat(False, 6)
        # self.solu = np.array([78.00000000000000000,       # old optimum from R
        #                       33.00000000000000000,
        #                       29.99525602568341398,
        #                       44.99999999847009491,
        #                       36.77581290640451783])
        self.solu = np.array([78.00000000000000000,         # slightly better optimum from cop.solve_G04(47),
                              33.00000000000000000,         # gives fully feasible solution and avoids 'negative errors'
                              29.99525602568163,
                              45.00000000000000000,
                              36.77581290578813])
        # self$x0=c(runif(1,min=78,max=102), #x1    # random point in R
        #           runif(1,min=33,max=45),  #x2
        #           runif(3,min=27,max=45))
        self.x0 = np.array([80,40,35,35,35])        # fixed choice for reproducible results
        self.fn = lambda x: np.array([(5.3578547*(x[2]**2))+(0.8356891*x[0]*x[4])+(37.293239*x[0])-40792.141,
                                      -(85.334407+0.0056858*x[1]*x[4]+0.0006262*x[0]*x[3]-0.0022053*x[2]*x[4]),
                                      85.334407+0.0056858*x[1]*x[4]+0.0006262*x[0]*x[3]-0.0022053*x[2]*x[4]-92,
                                      90-(80.51249+(0.0071317*x[1]*x[4])+(0.0029955*x[0]*x[1])+(0.0021813*x[2]**2)),
                                      80.51249+(0.0071317*x[1]*x[4])+(0.0029955*x[0]*x[1])+(0.0021813*x[2]**2)-110,
                                      20-(9.300961+(0.0047026*x[2]*x[4])+(0.0012547*x[0]*x[2])+(0.0019085*x[2]*x[3])),
                                      9.300961+(0.0047026*x[2]*x[4])+(0.0012547*x[0]*x[2])+(0.0019085*x[2]*x[3])-25])

    def _call_G05(self):
        self.dimension = 4
        self.lower = np.concatenate((np.repeat(0,2), np.repeat(-0.55, 2)))
        self.upper = np.concatenate((np.repeat(1200, 2), np.repeat(0.55, 2)))
        self.nConstraints = 5
        self.is_equ = np.array([False, False, True, True, True])
        self.solu = np.array([679.94531748791177961,
                              1026.06713513571594376,
                              0.11887636617838561,
                              -0.39623355240329272])
        # self.x0 = np.concatenate((np.random.rand(2) * 1200,             # x1, x2  # original: random start point
        #                          0.55 * (np.random.rand(2)*2 - 1)))     # x3, x4
        self.x0 = np.concatenate((np.array([0.4,0.6]) * 1200,             # x1, x2  # fixed choice for reproducible
                                 0.55 * (np.array([0.4,0.6])*2 - 1)))     # x3, x4  # results
        self.fn = lambda x: np.array([3*x[0]+1e-6*(x[0]**3)+2*x[1]+(2*1e-6/3)*(x[1]**3),
                                      x[2] - x[3] - 0.55,
                                      x[3] - x[2] - 0.55,
                                      1000 * np.sin(-x[2] - 0.25) + 1000 * np.sin(-x[3] - 0.25) + 894.8 - x[0],
                                      1000 * np.sin( x[2] - 0.25) + 1000 * np.sin( x[2] - x[3] - 0.25) + 894.8 - x[1],
                                      1000 * np.sin( x[3] - 0.25) + 1000 * np.sin( x[3] - x[2] - 0.25) + 1294.8])

    def _call_G06(self):
        self.dimension = 2
        self.lower = np.array([13,0])
        self.upper = np.array([100,100])
        self.nConstraints = 2
        self.is_equ = np.array([False, False])
        self.solu = np.array([14.095, 5 - np.sqrt(100 - (14.095 - 5) ** 2)])
        self.x0 = np.array([20.1, 5.84])
        self.fn = lambda x: np.array([((x[0] - 10) ** 3) + ((x[1] - 20) ** 3),
                                      -((((x[0] - 5) ** 2) + ((x[1] - 5) ** 2) - 100)),
                                      ((x[0] - 6) ** 2) + ((x[1] - 5) ** 2) - 82.81])

    def _call_G07(self):
        self.dimension = 10
        self.lower = np.repeat(-10, self.dimension)
        self.upper = np.repeat(+10, self.dimension)
        self.nConstraints = 8
        self.is_equ = np.repeat(False, self.dimension)
        self.solu = np.array([2.171997834812, 2.363679362798, 8.773925117415,
                              5.095984215855, 0.990655966387, 1.430578427576,
                              1.321647038816, 9.828728107011, 8.280094195305, 8.375923511901 ])
        self.x0 = None
        self.fn = lambda x: np.array([(x[0]**2)+(x[1]**2)+(x[0]*x[1])-(14*x[0])-(16*x[1])+((x[2]-10)**2)+
                                      (4*((x[3]-5)**2))+ ((x[4]-3)**2) +( 2*((x[5]-1)**2)) +(5*(x[6]**2))+
                                      ( 7*((x[7]-11)**2))+(2*((x[8]-10)**2 ))+ ((x[9]-7)**2) + 45,
                                      ((4.0*x[0])+(5.0*x[1])-(3.0*x[6])+(9.0*x[7]) - 105.0),        # g1
                                      (10*x[0]-8*x[1]-17*x[6]+2*x[7]),                              # g2
                                      (-8 * x[0] + 2 * x[1] + 5 * x[8] - 2 * x[9] - 12),            # g3
                                      (3 * ((x[0] - 2) ** 2) + 4 * (x[1] - 3) ** 2 + 2 * x[2] ** 2 - 7 * x[3] - 120),
                                      (5 * x[0] ** 2 + 8 * x[1] + (x[2] - 6) ** 2 - 2 * x[3] - 40),
                                      (x[0] ** 2 + 2 * (x[1] - 2) ** 2 - (2 * x[0] * x[1]) + 14 * x[4] - 6 * x[5]),
                                      (0.5 * (x[0] - 8) ** 2 + 2 * (x[1] - 4) ** 2 + 3 * (x[4] ** 2) - x[5] - 30),
                                      (-3 * x[0] + 6 * x[1] + 12 * (x[8] - 8) ** 2 - 7 * x[9])
                                    ])

    def _call_G11(self):
        self.dimension = 2
        self.lower = np.array([-1, -1])
        self.upper = np.array([1, 1])
        self.nConstraints = 1
        self.is_equ = np.array([True])
        self.solu = np.array([-np.sqrt(0.5), 0.5])
        self.fn = lambda x: np.array([x[0]**2 + (x[1]-1)**2,
                                      x[1] - x[0]**2 ])

    def _call_G13(self):
        self.dimension = 5
        self.lower = np.concatenate((np.repeat(-2.3,2),np.repeat(-3.2,3)))
        self.upper = np.concatenate((np.repeat(+2.3,2),np.repeat(+3.2,3)))
        self.nConstraints = 3
        self.is_equ = np.repeat(True, 3)
        self.fn = lambda x: np.array([np.exp(x[0]*x[1]*x[2]*x[3]*x[4]),
                                      x[0]**2+x[1]**2+x[2]**2+x[3]**2+x[4]**2-10,
                                      x[1] * x[2] - 5 * x[3] * x[4],
                                      x[0] ** 3 + x[1] ** 3 + 1
                                     ])
        solu0 = np.array([-1.7171435947203,1.5957097321519,1.8272456947885,-0.7636422812896,-0.7636439027742])
        self.solu = solu0.copy()
        self.solu = np.vstack((self.solu, np.array([solu0[0],solu0[1],-solu0[2],-solu0[3],+solu0[4]])))
        self.solu = np.vstack((self.solu, np.array([solu0[0],solu0[1],-solu0[2],+solu0[3],-solu0[4]])))
        self.solu = np.vstack((self.solu, np.array([solu0[0],solu0[1],+solu0[2],-solu0[3],-solu0[4]])))
        self.solu = np.vstack((self.solu, np.array([solu0[0],solu0[1],-solu0[2],+solu0[3],-solu0[4]])))
        self.solu = np.vstack((self.solu, np.array([solu0[0],solu0[1],-solu0[2],-solu0[3],+solu0[4]])))
        self.x0 = None
        self.info = "Please note that G13 has multiple global optima, all stored in solu"

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