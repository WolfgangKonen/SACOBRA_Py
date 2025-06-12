import numpy as np
import matplotlib.pyplot as plt
from cobraInit import CobraInitializer

class COP:
    """
        Abstract base class, just indicating certain members that a COP has to have
        (i.e. for calling functions like :meth:`show_error_plot`)
    """
    def __init__(self):
        self.name = ""
        self.fn = None
        self.lower = None
        self.upper = None
        self.fbest = None


class GCOP(COP):
    """
    Constraint Optimization Problem Benchmark (G Function Suite)

    [LiangRunar] J. Liang, T. P. Runarsson, E. Mezura-Montes, M. Clerc, P. Suganthan, C. C. Coello, and K. Deb, “Problem definitions and evaluation criteria for the CEC 2006 special session on constrained real-parameter optimization,” Journal of Applied Mechanics, vol. 41, p. 8, 2006. `http://www.lania.mx/~emezura/util/files/tr_cec06.pdf <http://www.lania.mx/~emezura/util/files/tr_cec06.pdf>`_

    Example: Instantiate problem G01 or G02 with

    - ``G01 = GCOP("G01")``.
    - ``G02 = GCOP("G02", dimension=5)``.

    Only problems G02 and G03 have the extra parameter ``dimension``.
    All other problems G01, G04, ..., G24 have fixed dimensions.

    Objects of class ``GCOP`` have the following useful attributes:

    - **name**      name of the problem, given by the user as 1st argument
    - **dimension** input space dimension of the problem. For the scalable problems ``G02`` and ``G03``, the dimension should be given by the user, otherwise it will be set automatically
    - **lower**     lower bound vector, length = input space dimension
    - **upper**     upper bound vector, length = input space dimension
    - **fn**        the COP functions which can be passed to **SACOBRA_Py** (see parameter ``fn`` in :ref:`class CobraInitializer <cobraInit-label>`).
    - **nConstraints** number of constraints
    - **x0**        the suggested optimization starting point, may be ``None`` if not available
    - **solu**      the best known solution(s), (only for diagnostics purposes). Can be ``None`` (not known) or a vector in case of a single solution or a matrix in case of multiple equivalent solutions (each row of the matrix is a solution)
    - **fbest**     the objective at the best known solution(s), (only for diagnostics purposes)
    - **info**      information about the problem, may be ``None`` if not available
    """
    #
    # *** TODO: check validity of G08, G09, G10, G16, G19 ***
    #
    def __init__(self, name, dimension=None):
        super().__init__()
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
        elif name == "G08": self._call_G08()
        elif name == "G09": self._call_G09()
        elif name == "G10": self._call_G10()
        elif name == "G11": self._call_G11()
        elif name == "G12": self._call_G12()
        elif name == "G13": self._call_G13()
        elif name == "G14": self._call_G14()
        elif name == "G15": self._call_G15()
        elif name == "G16": self._call_G16()
        elif name == "G17": self._call_G17()
        elif name == "G18": self._call_G18()
        elif name == "G19": self._call_G19()
        elif name == "G21": self._call_G21()
        else:
            strg = f"G-function {name} not (yet) implemented"
            raise NotImplementedError(strg)

        if self.solu is None:
            self.fbest = None
        else:
            first_solu = self.solu[0, :] if self.solu.ndim == 2 else self.solu
            self.fbest = self.fn(first_solu)[0]     # objective at (first) solution point

    def _call_G01(self):
        self.dimension = 13
        self.lower = np.repeat(0, self.dimension)
        self.upper = np.concatenate((np.repeat(1, 9), np.repeat(100, 3), [1]))
        self.nConstraints = 9
        self.is_equ = np.repeat(False, 9)
        self.solu = np.concatenate((np.repeat(1, 9), np.repeat(3, 3), [1]))
        self.x0 = np.repeat(0, self.dimension)
        self.fn = lambda x: np.array([np.sum(5*x[0:4])-(5*sum(x[0:4]*x[0:4]))-(sum(x[4:13])),
                                      (2*x[0]+2*x[1]+x[9]+x[10] - 10),
                                      (2*x[0]+2*x[2]+x[9]+x[11] - 10),
                                      (2*x[1]+2*x[2]+x[10]+x[11] - 10),
                                      -8*x[0]+x[9],
                                      -8*x[1]+x[10],
                                      -8*x[2]+x[11],
                                      -2*x[3]-x[4]+x[9],
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
                                              0.3633289, 0.3592627, 0.3547453, 0.3510025])
        else: self.solu = None
        # no x0 provided

        def denom(x):
            return np.sqrt(np.sum(np.array([(i+1)*x[i] for i in range(dim)])))
        self.fn = lambda x: np.array([-np.abs(np.sum(np.cos(x)**4)-(2*np.prod(np.cos(x)**2))/denom(x)),
                                      0.75 - np.prod(x),
                                      np.sum(x) - 7.5*dim])

    def _call_G03(self, dim):
        assert dim is not None, "[_call_G03] dimension has to be integer, not None"
        assert type(dim) is int, "[_call_G03] dimension has to be integer"
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
        self.x0 = np.array([80, 40, 35, 35, 35])    # fixed choice for reproducible results
        self.fn = lambda x: np.array([(5.3578547*(x[2]**2))+(0.8356891*x[0]*x[4])+(37.293239*x[0])-40792.141,
                                      -(85.334407+0.0056858*x[1]*x[4]+0.0006262*x[0]*x[3]-0.0022053*x[2]*x[4]),
                                      85.334407+0.0056858*x[1]*x[4]+0.0006262*x[0]*x[3]-0.0022053*x[2]*x[4]-92,
                                      90-(80.51249+(0.0071317*x[1]*x[4])+(0.0029955*x[0]*x[1])+(0.0021813*x[2]**2)),
                                      80.51249+(0.0071317*x[1]*x[4])+(0.0029955*x[0]*x[1])+(0.0021813*x[2]**2)-110,
                                      20-(9.300961+(0.0047026*x[2]*x[4])+(0.0012547*x[0]*x[2])+(0.0019085*x[2]*x[3])),
                                      9.300961+(0.0047026*x[2]*x[4])+(0.0012547*x[0]*x[2])+(0.0019085*x[2]*x[3])-25])

    def _call_G05(self):
        self.dimension = 4
        self.lower = np.concatenate((np.repeat(0, 2), np.repeat(-0.55, 2)))
        self.upper = np.concatenate((np.repeat(1200, 2), np.repeat(0.55, 2)))
        self.nConstraints = 5
        self.is_equ = np.array([False, False, True, True, True])
        self.solu = np.array([679.94531748791177961,
                              1026.06713513571594376,
                              0.11887636617838561,
                              -0.39623355240329272])
        # self.x0 = np.concatenate((np.random.rand(2) * 1200,             # x1, x2  # original: random start point
        #                          0.55 * (np.random.rand(2)*2 - 1)))     # x3, x4
        self.x0 = np.concatenate((np.array([0.4, 0.6]) * 1200,            # x1, x2  # fixed choice for reproducible
                                 0.55 * (np.array([0.4, 0.6])*2 - 1)))    # x3, x4  # results
        self.fn = lambda x: np.array([3*x[0]+1e-6*(x[0]**3)+2*x[1]+(2*1e-6/3)*(x[1]**3),
                                      x[2] - x[3] - 0.55,
                                      x[3] - x[2] - 0.55,
                                      1000 * np.sin(-x[2] - 0.25) + 1000 * np.sin(-x[3] - 0.25) + 894.8 - x[0],
                                      1000 * np.sin( x[2] - 0.25) + 1000 * np.sin( x[2] - x[3] - 0.25) + 894.8 - x[1],
                                      1000 * np.sin( x[3] - 0.25) + 1000 * np.sin( x[3] - x[2] - 0.25) + 1294.8])

    def _call_G06(self):
        self.dimension = 2
        self.lower = np.array([13, 0])
        self.upper = np.array([100, 100])
        self.nConstraints = 2
        self.is_equ = np.array([False, False])
        self.solu = np.array([14.095, 5 - np.sqrt(100 - (14.095 - 5) ** 2)])
        self.x0 = np.array([20.1, 5.84])
        self.fn = lambda x: np.array([((x[0] - 10) ** 3) + ((x[1] - 20) ** 3),
                                      -(((x[0] - 5) ** 2) + ((x[1] - 5) ** 2) - 100),
                                      ((x[0] - 6) ** 2) + ((x[1] - 5) ** 2) - 82.81])

    def _call_G07(self):
        self.dimension = 10
        self.lower = np.repeat(-10, self.dimension)
        self.upper = np.repeat(+10, self.dimension)
        self.nConstraints = 8
        self.is_equ = np.repeat(False, self.nConstraints)
        self.solu = np.array([2.171997834812, 2.363679362798, 8.773925117415,
                              5.095984215855, 0.990655966387, 1.430578427576,
                              1.321647038816, 9.828728107011, 8.280094195305, 8.375923511901])
        self.x0 = None
        self.fn = lambda x: np.array([(x[0]**2)+(x[1]**2)+(x[0]*x[1])-(14*x[0])-(16*x[1])+((x[2]-10)**2) +
                                      (4*((x[3]-5)**2))+ ((x[4]-3)**2) +(2*((x[5]-1)**2)) +(5*(x[6]**2)) +
                                      (7*((x[7]-11)**2))+(2*((x[8]-10)**2))+ ((x[9]-7)**2) + 45,
                                      (4.0*x[0])+(5.0*x[1])-(3.0*x[6])+(9.0*x[7]) - 105.0,        # g1
                                      10*x[0]-8*x[1]-17*x[6]+2*x[7],                              # g2
                                      -8 * x[0] + 2 * x[1] + 5 * x[8] - 2 * x[9] - 12,            # g3
                                      3 * ((x[0] - 2) ** 2) + 4 * (x[1] - 3) ** 2 + 2 * x[2] ** 2 - 7 * x[3] - 120,
                                      5 * x[0] ** 2 + 8 * x[1] + (x[2] - 6) ** 2 - 2 * x[3] - 40,
                                      x[0] ** 2 + 2 * (x[1] - 2) ** 2 - (2 * x[0] * x[1]) + 14 * x[4] - 6 * x[5],
                                      0.5 * (x[0] - 8) ** 2 + 2 * (x[1] - 4) ** 2 + 3 * (x[4] ** 2) - x[5] - 30,
                                      -3 * x[0] + 6 * x[1] + 12 * (x[8] - 8) ** 2 - 7 * x[9]
                                     ])

    def _call_G08(self):
        self.dimension = 2
        self.lower = np.repeat(0.00001, self.dimension)
        self.upper = np.repeat(+10, self.dimension)
        self.nConstraints = 2
        self.is_equ = np.repeat(False, self.nConstraints)
        self.solu = np.array([1.2279713,4.2453733])
        self.x0 = None
        self.fn = lambda x: np.array([((-np.sin(2*np.pi*x[0])**3)*(np.sin(2*np.pi*x[1]))) / ((x[0]**3)*(x[0]+x[1])),
                                      x[0]**2-x[1]+1,                        # g1
                                      1-x[0]+(x[1]-4)**2,                    # g2
                                     ])

    def _call_G09(self):
        self.dimension = 7
        self.lower = np.repeat(-10, self.dimension)
        self.upper = np.repeat(+10, self.dimension)
        self.nConstraints = 4
        self.is_equ = np.repeat(False, self.nConstraints)
        self.solu = np.array([2.33049949323300210,
                              1.95137239646596039,
                             -0.47754041766198602,
                              4.36572612852776931,
                             -0.62448707583702823,
                              1.03813092302119347,
                              1.59422663221959926])
        self.x0 = None
        self.fn = lambda x: np.array([(x[0]-10)**2+5*(x[1]-12)**2+x[2]**4+3*(x[3]-11)**2+10*(x[4]**6)+7*x[5]**2 +x[6]**4-4*x[5]*x[6]-10*x[5]-8*x[6],
                                      (2*x[0]**2+3*x[1]**4+x[2]+4*x[3]**2+5*x[4]-127),                        # g1
                                      (7*x[0]+3*x[1]+10*x[2]**2+x[3]-x[4]-282),                    # g2
                                      (23*x[0]+x[1]**2+6*x[5]**2-8*x[6]-196),  # g3
                                      4*x[0]**2+x[1]**2-3*x[0]*x[1]+2*x[2]**2+5*x[5]-11*x[6],  # g4
                                      ])

    def _call_G10(self):
        self.dimension = 8
        self.lower = np.array([100,1000,1000,np.repeat(10, 5)])
        self.upper = np.array([np.repeat(10000, 3), np.repeat(1000,5)])
        self.nConstraints = 6
        self.is_equ = np.repeat(False, self.nConstraints)
        self.solu = np.array([579.29340269759155,
                              1359.97691009458777,
                              5109.97770901501008,
                              182.01659025342749,
                              295.60089166064103,
                              217.98340973906758,
                              286.41569858295981,
                              395.60089165381908])
        self.x0 = np.array([np.repeat(1001, 3), np.repeat(100,self.dimension-3)])
        self.fn = lambda x: np.array([x[0]+x[1]+x[2],
                                      (-1+0.0025*(x[3]+x[5])),                          # g1
                                      (-1 + 0.0025*(-x[3]+x[4]+x[7])),                  # g2
                                      (-1+0.01*(-x[4]+x[7])),                           # g3
                                      (100*x[0]-(x[0]*x[5])+833.33252*x[3]-83333.333),  # g4
                                      (x[1]*x[3]-x[1]*x[7]-1250*x[3]+1250*x[4]),        # g5
                                      (x[2]*x[4]-x[2]*x[7]-2500*x[4]+1250000),          # g6
                                      ])

    def _call_G11(self):
        self.dimension = 2
        self.lower = np.array([-1, -1])
        self.upper = np.array([1, 1])
        self.nConstraints = 1
        self.is_equ = np.array([True])
        self.solu = np.array([-np.sqrt(0.5), 0.5])
        self.fn = lambda x: np.array([x[0]**2 + (x[1]-1)**2,
                                      x[1] - x[0]**2])


    def _call_G12(self):
        self.dimension = 3
        self.lower = np.repeat(0, 3)
        self.upper = np.repeat(10, 3)
        self.nConstraints = 1
        self.is_equ = np.array([False])
        G = lambda x: np.min(np.array([[[(x[0]-i)**2+(x[1]-j)**2+(x[2]-k)**2 - 0.0625
                                      for i in range(1,9)] for j in range(1,9)] for k in range(1,9)]))
        # np.array contains 729 disjoint spheres of radius 0.25. A solution is feasible if it is within *one* of
        # the 729 spheres. Therefore, we take the min over the array to obtain the single constraint G.
        self.fn = lambda x: np.array([-1 + 0.01*((x[0]-5)**2+(x[1]-5)**2+(x[2]-5)**2),
                                      G(x) ])
        self.solu = np.array([5., 5., 5.])
        self.x0 = None

    def _call_G13(self):
        self.dimension = 5
        self.lower = np.concatenate((np.repeat(-2.3, 2), np.repeat(-3.2, 3)))
        self.upper = np.concatenate((np.repeat(+2.3, 2), np.repeat(+3.2, 3)))
        self.nConstraints = 3
        self.is_equ = np.repeat(True, 3)
        self.fn = lambda x: np.array([np.exp(x[0]*x[1]*x[2]*x[3]*x[4]),
                                      x[0]**2+x[1]**2+x[2]**2+x[3]**2+x[4]**2-10,
                                      x[1] * x[2] - 5 * x[3] * x[4],
                                      x[0] ** 3 + x[1] ** 3 + 1
                                     ])
        solu0 = np.array([-1.7171435947203, 1.5957097321519, 1.8272456947885, -0.7636422812896, -0.7636439027742])
        self.solu = solu0.copy()
        self.solu = np.vstack((self.solu, np.array([solu0[0], solu0[1], -solu0[2], -solu0[3], +solu0[4]])))
        self.solu = np.vstack((self.solu, np.array([solu0[0], solu0[1], -solu0[2], +solu0[3], -solu0[4]])))
        self.solu = np.vstack((self.solu, np.array([solu0[0], solu0[1], +solu0[2], -solu0[3], -solu0[4]])))
        self.solu = np.vstack((self.solu, np.array([solu0[0], solu0[1], -solu0[2], +solu0[3], -solu0[4]])))
        self.solu = np.vstack((self.solu, np.array([solu0[0], solu0[1], -solu0[2], -solu0[3], +solu0[4]])))
        self.x0 = None
        self.info = "Please note that G13 has multiple global optima, all stored in solu"

    def _call_G14(self):
        self.dimension = 10
        self.lower = np.repeat(1e-6, 10)
        self.upper = np.repeat(10, 10)
        self.nConstraints = 3
        self.is_equ = np.repeat(True, 3)
        cvec = np.array([-6.089, -17.164, -34.054, -5.914, -24.721, -14.986, -24.1, -10.708, -26.662, -22.179])
        self.fn = lambda x: np.array([np.sum(x*(cvec+np.log(x/np.sum(x)))),
                                      x[0]+2*x[1]+2*x[2]+x[5]+x[9]-2,
                                      x[3]+2*x[4]+x[5]+x[6]-1,
                                      x[2]+x[6]+x[7]+2*x[8]+x[9]-1
                                      ])
        self.solu = np.array([0.0406684113216282, 0.147721240492452, 0.783205732104114, 0.00141433931889084,
                              0.485293636780388, 0.000693183051556082, 0.0274052040687766, 0.0179509660214818,
                              0.0373268186859717, 0.0968844604336845])
        self.x0 = None

    def _call_G15(self):
        self.dimension = 3
        self.lower = np.repeat(0, 3)
        self.upper = np.repeat(10, 3)
        self.nConstraints = 2
        self.is_equ = np.repeat(True, 2)
        self.fn = lambda x: np.array([1000-(x[0]**2)-2*x[1]**2-x[2]**2-x[0]*x[1]-x[0]*x[2],
                                      x[0]**2+x[1]**2+x[2]**2-25,
                                      8*x[0]+14*x[1]+7*x[2]-56
                                      ])
        self.solu = np.array([3.51212812611795133, 0.216987510429556135, 3.55217854929179921])
        self.x0 = None

    def _call_G16(self):
        self.dimension = 5
        self.lower = np.array([704.4148,68.6,0,193,25])
        self.upper = np.array([906.3855,288.88,134.75,287.0966,84.1988])
        self.nConstraints = 38
        self.is_equ = np.repeat(False, self.nConstraints)

        def func_fn(x):
            y1 = x[1] + x[2] + 41.6
            c1 = 0.024 * x[3]-4.62
            y2 = (12.5 / c1) + 12
            c2 = 0.0003535 * (x[0] ** 2) + 0.5311 * x[0] + 0.08705 * y2 * x[0]
            c3 = 0.052 * x[0] + 78 + 0.002377 * y2 * x[0]
            y3 = c2 / c3
            y4 = 19 * y3
            c4 = 0.04782 * (x[0] - y3) + (0.1956 * (x[0] - y3) ** 2) / x[1] + 0.6376 * y4 + 1.594 * y3
            c5 = 100 * x[1]
            c6 = x[0] - y3 - y4
            c7 = 0.950 - (c4 / c5)
            y5 = c6 * c7
            y6 = x[0] - y5 - y4 - y3
            c8 = (y5 + y4) * 0.995
            y7 = c8 / y1
            y8 = c8 / 3798
            c9 = y7 - (0.0663 * (y7 / y8)) - 0.3153
            y9 = (96.82 / c9) + 0.321 * y1
            y10 = 1.29 * y5 + 1.258 * y4 + 2.29 * y3 + 1.71 * y6
            y11 = 1.71 * x[0] - 0.452 * y4 + 0.580 * y3
            c10 = 12.3 / 752.3
            c11 = (1.75 * y2) * (0.995 * x[0])
            c12 = (0.995 * y10) + 1998
            y12 = c10 * x[0] + (c11 / c12)
            y13 = c12 - 1.75 * y2
            y14 = 3623 + 64.4 * x[1] + 58.4 * x[2] + 146312 / (y9 + x[4])
            c13 = 0.995 * y10 + 60.8 * x[1] + 48 * x[3] - 0.1121 * y14 - 5095
            y15 = y13 / c13
            y16 = 148000 - 331000 * y15 + 40 * y13 - 61 * y15 * y13
            c14 = 2324 * y10 - 28740000 * y2
            y17 = 14130000 - (1328 * y10) - (531 * y11) + (c14 / c12)
            c15 = (y13 / y15) - (y13 / 0.52)
            c16 = 1.104 - 0.72 * y15
            c17 = y9 + x[4]

            obj = ((0.000117 * y14) + 0.1365 + (0.00002358 * y13) + (0.000001502 * y16) + (0.0321 * y12) +
                   (0.004324 * y5) + (0.0001 * c15 / c16) + (37.48 * (y2 / c12)) - (0.0000005843 * y17))
            g1 =(0.28 / 0.72) * y5 - y4
            g2 = x[3] - 1.5 * x[2]
            g3 = 3496 * (y2 / c12) - 21
            g4 = 110.6 + y1 - (62212 / c17)
            g5 = 213.1 - y1
            g6 = y1 - 405.23
            g7 = 17.505 - y2
            g8 = y2 - 1053.6667
            g9 = 11.275 - y3
            g10 = y3 - 35.03
            g11 = 214.228 - y4
            g12 = y4 - 665.585
            g13 = 7.458 - y5
            g14 = y5 - 584.463
            g15 = 0.961 - y6
            g16 = y6 - 265.916
            g17 = 1.612 - y7
            g18 = y7 - 7.046
            g19 = 0.146 - y8
            g20 = y8 - 0.222
            g21 = 107.99 - y9
            g22 = y9 - 273.366
            g23 = 922.693 - y10
            g24 = y10 - 1286.105
            g25 = 926.832 - y11
            g26 = y11 - 1444.046
            g27 = 18.766 - y12
            g28 = y12 - 537.141
            g29 = 1072.163 - y13
            g30 = y13 - 3247.039
            g31 = 8961.448 - y14
            g32 = y14 - 26844.086
            g33 = 0.063 - y15
            g34 = y15 - 0.386
            g35 = 71084.33 - y16
            g36 = -140000 + y16
            g37 = 2802713 - y17
            g38 = y17 - 12146108

            return np.array([ obj
                             , g1, g2, g3, g4, g5, g6, g7, g8, g9
                             , g10, g11, g12, g13, g14, g15, g16, g17, g18
                             , g19, g20, g21, g22, g23, g24, g25, g26, g27
                             , g28, g29, g30, g31, g32, g33, g34, g35, g36, g37, g38])

        self.fn = lambda x: func_fn(x)
        self.solu = np.array([705.17454,  68.60000, 102.90000, 282.32493,  37.58412])
        self.x0 = None



    def _call_G17(self):
        self.dimension = 6
        self.lower = np.array([0, 0, 340, 340, -1000, 0.0])
        self.upper = np.array([400, 1000, 420, 420, 1000, 0.5236])
        self.nConstraints = 4
        self.is_equ = np.repeat(True, 4)
        f0 = lambda x0: 30*x0 if 0 <= x0 < 300 else 31 * x0
        f2 = lambda x1: 29*x1 if 100 <= x1 < 200 else 30 * x1
        f1 = lambda x1: 28*x1 if 0 <= x1 < 100 else f2(x1)
        self.fn = lambda x: np.array([f0(x[0]) + f1(x[1]),
                                      -x[0] + 300 - (x[2] * x[3]) / 131.078 * np.cos(1.48477 - x[5]) +
                                                  (0.90798 * x[2] ** 2) / 131.078 * np.cos(1.47588),
                                      -x[1] - (x[2] * x[3]) / 131.078 * np.cos(1.48477 + x[5]) +
                                                  (0.90798 * x[3] ** 2) / 131.078 * np.cos(1.47588),
                                      -x[4] - (x[2] * x[3]) / 131.078 * np.sin(1.48477 + x[5]) +
                                                  (0.90798 * x[3] ** 2) / 131.078 * np.sin(1.47588),
                                      200 - (x[2] * x[3]) / 131.078 * np.sin(1.48477 - x[5]) +
                                                  (0.90798 * x[2] ** 2) / 131.078 * np.sin(1.47588)
                                    ])
        self.solu = np.array([201.784467214523659, 99.9999999999999005, 383.071034852773266,
                              420, -10.9076584514292652, 0.0731482312084287128])
        self.x0 = None

    def _call_G18(self):
        self.dimension = 9
        self.lower = np.concatenate((np.repeat(-10,8), [0]))
        self.upper = np.concatenate((np.repeat(+10,8), [20]))
        self.nConstraints = 13
        self.is_equ = np.repeat(False, 13)
        self.fn = lambda x: np.array([-0.5*(x[0]*x[3]-x[1]*x[2] + x[2]*x[8]-x[4]*x[8] + x[4]*x[7]- x[5]*x[6]),
                                      x[2] ** 2 + x[3] ** 2 - 1,
                                      x[8] ** 2 - 1,
                                      x[4] ** 2 + x[5] ** 2 - 1,
                                      x[0] ** 2 + (x[1] - x[8]) ** 2 - 1,
                                      (x[0] - x[4]) ** 2 + (x[1] - x[5]) ** 2 - 1,
                                      (x[0] - x[6]) ** 2 + (x[1] - x[7]) ** 2 - 1,
                                      (x[2] - x[4]) ** 2 + (x[3] - x[5]) ** 2 - 1,
                                      (x[2] - x[6]) ** 2 + (x[3] - x[7]) ** 2 - 1,
                                      x[6] ** 2 + (x[7] - x[8]) ** 2 - 1,
                                      x[1] * x[2] - x[0] * x[3],
                                      -x[2] * x[8],
                                      x[4] * x[8],
                                      x[5] * x[6] - x[4] * x[7]
                                      ])
        self.solu = np.array([-0.9890005492667746,0.1479118418638228,
                              -0.6242897641574451,-0.7811841737429015,
                              -0.9876159387318453,0.1504778305249072,
                              -0.6225959783340022,-0.782543417629948, 0.0])
        self.x0 = None

    def _call_G19(self):
        self.dimension = 15
        self.lower = np.repeat(0, self.dimension)
        self.upper = np.repeat(10, self.dimension)
        self.nConstraints = 5
        self.is_equ = np.repeat(False, self.nConstraints)
        self.aMat19 = np.array([
                       -16 ,  2,  0,  1,   0,
                       +0  , -2,  0,0.4,   2,
                       -3.5,  0,  2,  0,   0,
                       +0  , -2,  0, -4,  -1,
                       +0  , -9, -2,  1,-2.8,
                       +2  ,  0, -4,  0,   0,
                       -1  , -1, -1, -1,  -1,
                       -1  , -2, -3, -2,  -1,
                       +1  ,  2,  3,  4,   5,
                       +1  ,  1,  1,  1,   1 ]).reshape((10,5))
        self.bVec19 = np.array([-40,-2,-0.25,-4,-4,-1,-40,-60,5,1])
        self.cMat19 = np.array([
                       +30, -20, -10, 32, -10,
                       -20,  39,  -6,-31,  32,
                       -10,  -6,  10, -6, -10,
                       +32, -31,  -6, 39, -20,
                       -10,  32, -10,-20,  30]).reshape((5,5))
        self.dVec19 = np.array([4, 8, 10, 6, 2])
        self.eVec19 = np.array([-15, -27, -36, -18, -12])

        def fitFunc(x):
            obj = -np.dot(self.bVec19, x[0:10]) + 2 * np.sum(self.dVec19 * x[10:15] * x[10:15] * x[10:15])
            for i in range(5):
                obj = obj + x[10 + i] * np.dot(self.cMat19[i,:], x[10:15])
            return obj

        def conFunc(x):
            res = np.zeros(5)
            for j in range(5):
                res[j] = (-2 * np.dot(self.cMat19[:, j], x[10:15]) -3 * self.dVec19[j] * x[10 + j] * x[10 + j]
                          - self.eVec19[j] + np.dot(self.aMat19[:,j], x[0:10]))
            return res

        self.fn = lambda x: [fitFunc(x)] + conFunc(x).tolist()
        self.solu = np.array([
            0, 0,  3.94600628013917,  0,    3.28318162727873, 10, 0, 0, 0,0,
            0.370762125835098, 0.278454209512692, 0.523838440499861, 0.388621589976956, 0.29815843730292])
        self.x0 = None

    def _call_G21(self):
        self.dimension = 7
        self.lower = np.array([0, 0, 0, 100, 6.3, 5.9, 4.5])
        self.upper = np.array([1000, 40, 40, 300, 6.7, 6.4, 6.25])
        self.nConstraints = 6
        self.is_equ = np.array([False, True, True, True, True, True])
        self.fn = lambda x: np.array([+x[0],     # obj
                                      -x[0]+35*(x[1]**(0.6))+35*(x[2]**0.6),                                       # g1
                                      -300*x[2] + 7500*x[4]- 7500*x[5] - 25*x[3]*x[4] + 25*x[3]*x[5] + x[2]*x[3],  # h1
                                      +100*x[1] + 155.365*x[3] + 2500*x[6] - x[1]*x[3] - 25*x[3]*x[6] - 15536.5,   # h2
                                      -x[4] + np.log(-x[3] + 900),                                                 # h3
                                      -x[5] + np.log(x[3] + 300),                                                  # h4
                                      -x[6] + np.log(-2 * x[3] + 700)                                              # h5
                                      ])
        self.solu = np.array([193.724510070034967, 5.56944131553368433e-27, 17.3191887294084914,
                              100.047897801386839, 6.68445185362377892, 5.99168428444264833, 6.21451648886070451])
        self.x0 = None


def show_error_plot(cobra: CobraInitializer, cop: COP, ylim=None, file=None):
    err = cobra.sac_res['fbestArray'] - cop.fbest
    plt.plot(range(err.size), np.abs(err), 'r-', label='error')
    plt.title(cop.name, fontsize=20)
    plt.xlabel('func evals ', fontsize=16)
    plt.ylabel('error', fontsize=16)
    plt.subplot(111).set_yscale("log")
    if ylim is not None:
        plt.subplot(111).set(ylim=ylim)
    if file is not None:
        plt.savefig(file)
    plt.show()


if __name__ == '__main__':
    G01 = GCOP("G01")
    print(G01.solu)
    print(G01.fn(G01.solu))
    G02 = GCOP("G02", 10)
    print(G02.solu)
    print(G02.fn(G02.solu))
    G15 = GCOP("G15")
    print(G15.solu)
    print(G15.fn(G15.solu))
