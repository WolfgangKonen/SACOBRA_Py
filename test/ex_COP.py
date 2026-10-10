import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from cobraInit import CobraInitializer
from gCOP import GCOP, show_error_plot
from cobraPhaseII import CobraPhaseII
from innerFuncs import distLine
from opt.equOptions import EQUoptions
from opt.isaOptions import ISAoptions, O_LOGIC
from opt.riOptions import RIoptions
from opt.sacOptions import SACoptions
from opt.idOptions import IDoptions
from opt.rbfOptions import RBFoptions
from opt.seqOptions import SEQoptions

verb = 1


def set_idp(dim, deg):
    if deg == 1:
        idp = dim + 1
    elif deg == 1.5:
        idp = 2 * dim + 1
    else:  # deg == 2
        idp = (dim + 1) * (dim + 2) // 2
    return idp


class ExamCOP:
    """
        Example COPs from the G function benchmark. Test for statistical equivalence to the R side (ex_COP.R).
        The class methods ``solve_Gxx`` allow to set specific parameters for each G function (used by
        :meth:`.OneS.one_s_multi_g_r` in conjunction with ``meth='solve'``).

        - G01 is a COP with 9 linear inequality constraints and d=13
        - G02 is a COP with 2 inequality constraints and steerable dimension d.
        - G03 is a COP with 1 equality constraint (sphere) and steerable dimension d.
        - G04 is a COP with 6 inequality constraints and d=5.
        - G05 is a COP with 2 inequality and 3 equality constraints. d=4.
        - G06 is a COP with two circular inequality constraints that form a very narrow feasible region. d=2.
        - G07 is a COP with 8 inequality constraints. d=10.
        - G08 is a COP with 2 inequality constraints and d=2.
        - G09 is a COP with 4 inequality constraints and d=7.
        - G10 is a COP with 6 inequality constraints and d=8.
        - G11 is a COP with 1 equality constraint. d=2.
        - G12 is a COP with 1 inequality constraint and d=3.
        - G13 is a COP with 3 equality constraints. d=5.
        - G14 is a COP with 3 equality constraints. d=10.
        - G15 is a COP with 5 equality constraints. d=3.
        - G17 is a COP with 4 equality constraints. d=6.
        - G18 is a COP with 13 inequality constraints and d=9.
        - G19 is a COP with 5 inequality constraints and d=15.
        - G20 is not yet implemented.
        - G21 is a COP with 5 equality constraints. d=7.
        - G22 is a COP with 19 equality constraints. d=2.
        - G23 is a COP with 4 equality constraints. d=9.
        - G24 is a COP with 2 inequality constraints and d=2.

        In summary, there are 10 COPs (G03, G05, G11, G13, G14, G15, G17, G21, G22, G23) with equality constraints.
    """

    def solve_G01(self, cobraSeed, feval=170, verbIter=10, conTol=0):
        """ Test whether COP G01 has statistical similar results to the R side with squares=T, if we set on the
            Python side RBF.degree=2 (which is similar, but not the same).

            We test that the median of 15 final errors is < 5e-6, which is statistically similar to the R side
            (see ex_COP.R)
        """
        print(f"Starting solve_G01({cobraSeed}) ...")
        G01 = GCOP("G01")
        idp = 105   # =(d+1)(d+2)/2, the minimum for RBF.kernel="cubic", RBF.degree=2 and d=13

        cobra = CobraInitializer(G01.x0, G01.fn, G01.name, G01.lower, G01.upper, G01.is_equ,
                                 solu=G01.solu,
                                 s_opts=SACoptions(verbose=verb, verboseIter=verbIter, feval=feval, cobraSeed=cobraSeed,
                                                   # saveIntermediate=True,
                                                   ID=IDoptions(initDesign="RAND_REP", initDesPoints=idp),
                                                   RBF=RBFoptions(degree=1),  # , kernel="gaussian"
                                                   SEQ=SEQoptions(finalEpsXiZero=False, conTol=conTol)))
        c2 = CobraPhaseII(cobra).start(gcop=G01)

        print(f"final err: {c2.p2.fin_err}")
        print(c2.p2.f_solu)
        print(c2.cobra.get_fbest())
        c2.p2.fe_thresh = 5e-6      # same accuracy 1.1e-6 for s_opts.SEQ.finalEpsXiZero=True or False
        return c2

    def solve_G02(self, cobraSeed, dimension=5, feval=350, verbIter=100, conTol=0):      # conTol=0 | 1e-7
        """ Test whether COP G02 has statistical similar results to the R side with squares=T, if we set on the
            Python side RBF.degree=2 (which is similar, but not the same).

            We test that the median of 15 final errors is < 1e-9, which is statistically better than the R side
            (see ex_COP.R)
        """
        print(f"Starting solve_G02({cobraSeed}, dim={dimension}, ...) ...")
        G02 = GCOP("G02", dimension)

        # x0 = np.arange(dimension)/dimension    # fixed x0
        x0 = G02.solu - 0.001       # G02.x0
        cobra = CobraInitializer(x0, G02.fn, G02.name, G02.lower, G02.upper, G02.is_equ,
                                 solu=G02.solu,
                                 s_opts=SACoptions(verbose=verb, verboseIter=verbIter, feval=feval, cobraSeed=cobraSeed,
                                                   ID=IDoptions(initDesign="LHS", rescale=True),
                                                   RBF=RBFoptions(degree=2, rho=2.5, rhoDec=2.0),
                                                   SEQ=SEQoptions(finalEpsXiZero=True, conTol=conTol)))
        print(f"idp = {cobra.sac_opts.ID.initDesPoints}")
        c2 = CobraPhaseII(cobra).start(gcop=G02)

        print(f"final err: {c2.p2.fin_err}")
        print(c2.p2.f_solu)
        print(c2.cobra.get_fbest())
        c2.p2.fe_thresh = 1e-9
        return c2

    def solve_G03(self, cobraSeed, dimension=8, feval=150, verbIter=10, conTol=0):      # conTol=0 | 1e-7
        """ Test whether COP G03 has statistical similar results to the R side with squares=T, if we set on the
            Python side RBF.degree=2 (which is similar, but not the same).

            We test that the median of 15 final errors is < 1e-9, which is statistically better than the R side
            (see ex_COP.R)
        """
        print(f"Starting solve_G03({cobraSeed}, dim={dimension}, ...) ...")
        muFinal = 1e-4   # 1e-4 | 1e-7     # before 2026/10/03: muFinal=1e-12 --> strange png_err_plot (colors)
        G03 = GCOP("G03", dimension, mu=muFinal)

        x0 = G03.x0             # None --> a random x0 will be set
        # x0 = np.arange(dimension)/dimension    # fixed x0
        dim = G03.dimension
        idp = (dim + 1) * (dim + 2) // 2
        if feval == 0: feval = idp + 2
        equ_opt = EQUoptions(muGrow=100, muDec=1.6, muFinal=muFinal, refinePrint=False, refineAlgo="L-BFGS-B")  # "BFGS_1"
        cobra = CobraInitializer(x0, G03.fn, G03.name, G03.lower, G03.upper, G03.is_equ,
                                 solu=G03.solu,
                                 s_opts=SACoptions(verbose=verb, verboseIter=verbIter, feval=feval, cobraSeed=cobraSeed,
                                                   ID=IDoptions(initDesign="LHS", initDesPoints=idp, rescale=True),  #
                                                   RBF=RBFoptions(degree=2),    # , rho=0.0, rhoDec=2.0
                                                   ISA=ISAoptions(onlinePLOG=O_LOGIC.MIDPTS),
                                                   EQU=equ_opt,
                                                   RI=RIoptions(repairInfeas=True, eps2=0, q=3, repairMargin=np.inf, checkIt=False), # new 2025/10/15
                                                   SEQ=SEQoptions(finalEpsXiZero=True, trueFuncForSurrogates=False, conTol=conTol)))
        print(f"idp = {cobra.sac_opts.ID.initDesPoints}")
        c2 = CobraPhaseII(cobra).start(gcop=G03)

        print(f"final err: {c2.p2.fin_err}")
        print(c2.p2.f_solu)
        print(c2.cobra.get_fbest())
        c2.p2.fe_thresh = 1e-9
        return c2

    def solve_G04(self, cobraSeed, feval=170, verbIter=10, conTol=0):       # conTol=0 | 1e-7
        """ Test whether COP G04 has statistical similar results to the R side with squares=T, if we set on the
            Python side RBF.degree=2 (which is similar, but not the same).

            We test that the median of 15 final errors is < 1e-9 (actually 6e-11 for rescale=False and 2e-10 for
            rescale=True), which is statistically better than the R side (see ex_COP.R)
        """
        print(f"Starting solve_G04({cobraSeed}) ...")
        G04 = GCOP("G04")

        equ_opt = EQUoptions(muGrow=100, muDec=1.6, muFinal=1e-7, refineAlgo="BFGS_0", refinePrint=False)
        cobra = CobraInitializer(G04.x0, G04.fn, G04.name, G04.lower, G04.upper, G04.is_equ,
                                 solu=G04.solu,             # /WK/ bug fix: this was missing before 2026/10/03
                                 s_opts=SACoptions(verbose=verb, verboseIter=verbIter, feval=feval, cobraSeed=cobraSeed,
                                                   ID=IDoptions(initDesign="LHS", rescale=False),
                                                   RBF=RBFoptions(degree=2),
                                                   EQU=equ_opt,
                                                   SEQ=SEQoptions(finalEpsXiZero=True, conTol=conTol)))
        c2 = CobraPhaseII(cobra).start(gcop=G04)

        print(f"final err: {c2.p2.fin_err}")
        print(c2.p2.f_solu)
        print(c2.cobra.get_fbest())
        c2.p2.fe_thresh = 1e-9
        return c2

    def solve_G05(self, cobraSeed, feval=170, verbIter=10, conTol=0):       # conTol=0 | 1e-7
        """ Test whether COP G05 has statistical similar results to the R side with squares=T, if we set on the
            Python side RBF.degree=2 (which is similar, but not the same).

            We test that the median of 15 final errors is < 5e-6, which is statistically similar to the R side
            (see ex_COP.R)
        """
        print(f"Starting solve_G05({cobraSeed}) ...")
        muFinal = 1e-4   # 1e-4 | 1e-7
        G05 = GCOP("G05", mu=muFinal)
        idp = 15   # =(d+1)(d+2)/2, the minimum for RBF.kernel="cubic", RBF.degree=2 and d=4

        cobra = CobraInitializer(G05.x0, G05.fn, G05.name, G05.lower, G05.upper, G05.is_equ,
                                 solu=G05.solu,
                                 s_opts=SACoptions(verbose=verb, verboseIter=verbIter, feval=feval, cobraSeed=cobraSeed,
                                                   ID=IDoptions(initDesign="LHS", initDesPoints=idp),
                                                   RBF=RBFoptions(degree=2),
                                                   EQU=EQUoptions(muDec=1.6, muFinal=muFinal, refinePrint=False,  # before 2026/10/03: muFinal=1e-12, never feasible
                                                                  refineAlgo="COBYLA"),  # "L-BFGS-B COBYLA"
                                                   SEQ=SEQoptions(finalEpsXiZero=True, conTol=conTol)))
        c2 = CobraPhaseII(cobra).start(gcop=G05)

        print(f"final err: {c2.p2.fin_err}")
        print(c2.p2.f_solu)
        print(c2.cobra.get_fbest())
        c2.p2.fe_thresh = 5e-6
        return c2

    def solve_G06(self, cobraSeed, feval=40, verbIter=10, conTol=0):        # conTol=0 | 1e-7
        """ Test whether COP G06 has statistical equivalent results to the R side with squares=T, if we set on the
            Python side RBF.degree=2 (which is similar, but not the same).

            We test that the median of 15 final errors is < 5e-6, which is statistically equivalent to the R side
            (see ex_COP.R, function multi_G06)
        """
        print(f"Starting solve_G06({cobraSeed}) ...")
        G06 = GCOP("G06")

        cobra = CobraInitializer(G06.x0, G06.fn, G06.name, G06.lower, G06.upper, G06.is_equ,
                                 solu=G06.solu,
                                 s_opts=SACoptions(verbose=verb, verboseIter=verbIter, feval=feval, cobraSeed=cobraSeed,
                                                   ID=IDoptions(initDesign="RAND_REP", initDesPoints=6),
                                                   RBF=RBFoptions(degree=2),
                                                   SEQ=SEQoptions(finalEpsXiZero=True, conTol=conTol)))

        c2 = CobraPhaseII(cobra).start(gcop=G06)

        # show_error_plot(cobra, G06, c2.get_muVec())

        print(f"final err: {c2.p2.fin_err}")
        print(c2.p2.f_solu)
        print(c2.cobra.get_fbest())
        # c2.p2.fe_thresh = 5e-6    # this is for s_opts.SEQ.finalEpsXiZero=False
        c2.p2.fe_thresh = 5e-8      # this is for s_opts.SEQ.finalEpsXiZero=True and s_opts.SEQ.conTol=1e-7
        return c2

    def solve_G07(self, cobraSeed, feval=180, verbIter=10, conTol=0):       # conTol=0 | 1e-7
        """ Test whether COP G07 has statistical equivalent results to the R side with squares=T, if we set on the
            Python side RBF.degree=2 (which is similar, but not the same).

            We test that the median of 15 final errors is < 5e-6, which is statistically equivalent to the R side
            (see ex_COP.R, function solve_G07, multi_gfnc)
        """
        print(f"Starting solve_G07({cobraSeed}) ...")
        G07 = GCOP("G07")
        idp = 11*12//2

        cobra = CobraInitializer(G07.x0, G07.fn, G07.name, G07.lower, G07.upper, G07.is_equ,
                                 solu=G07.solu,
                                 s_opts=SACoptions(verbose=verb, verboseIter=verbIter, feval=feval, cobraSeed=cobraSeed,
                                                   ID=IDoptions(initDesign="LHS", initDesPoints=idp),
                                                   RBF=RBFoptions(degree=2),
                                                   SEQ=SEQoptions(finalEpsXiZero=True, conTol=conTol)))

        c2 = CobraPhaseII(cobra).start(gcop=G07)

        print(f"final err: {c2.p2.fin_err}")
        print(c2.p2.f_solu)
        print(c2.cobra.get_fbest())
        c2.p2.fe_thresh = 1e-9
        return c2

    def solve_G08(self, cobraSeed, feval=180, verbIter=10, conTol=0):       # conTol=0 | 1e-7
        """ Test whether COP G08 has statistical equivalent results to the R side with squares=T, if we set on the
            Python side RBF.degree=2 (which is similar, but not the same).

            We test that the median of 15 final errors is < 5e-6, which is statistically equivalent to the R side
            (see ex_COP.R, function solve_G07, multi_gfnc)
        """
        print(f"Starting solve_G08({cobraSeed}) ...")
        G08 = GCOP("G08")
        idp = 3*4//2

        cobra = CobraInitializer(G08.x0, G08.fn, G08.name, G08.lower, G08.upper, G08.is_equ,
                                 solu=G08.solu,
                                 s_opts=SACoptions(verbose=verb, verboseIter=verbIter, feval=feval, cobraSeed=cobraSeed,
                                                   ID=IDoptions(initDesign="LHS", initDesPoints=idp),
                                                   RBF=RBFoptions(degree=2),
                                                   SEQ=SEQoptions(finalEpsXiZero=True, conTol=conTol)))

        c2 = CobraPhaseII(cobra).start(gcop=G08)

        print(f"final err: {c2.p2.fin_err}")
        print(c2.p2.f_solu)
        print(c2.cobra.get_fbest())
        c2.p2.fe_thresh = 1e-9
        return c2

    def solve_G09(self, cobraSeed, feval=500, verbIter=50, conTol=1e-7):        # conTol=0 | 1e-7
        """ Test whether COP G09 has statistical equivalent results to the R side with squares=T, if we set on the
            Python side RBF.degree=2 (which is similar, but not the same).

            We test that the median of 15 final errors is < 5e-6, which is statistically equivalent to the R side
            (see ex_COP.R, function solve_G07, multi_gfnc)
        """
        print(f"Starting solve_G09({cobraSeed}) ...")
        G09 = GCOP("G09")
        idp = 10*11//2
        # G09.x0 = G09.solu + 0.01

        cobra = CobraInitializer(G09.x0, G09.fn, G09.name, G09.lower, G09.upper, G09.is_equ,
                                 solu=G09.solu,
                                 s_opts=SACoptions(verbose=verb, verboseIter=verbIter, feval=feval, cobraSeed=cobraSeed,
                                                   ID=IDoptions(initDesign="LHS", initDesPoints=idp),
                                                   RBF=RBFoptions(degree=2),
                                                   SEQ=SEQoptions(finalEpsXiZero=True, trueFuncForSurrogates=False, conTol=conTol)))

        c2 = CobraPhaseII(cobra).start(gcop=G09)

        print(f"final err: {c2.p2.fin_err}")
        print(c2.p2.f_solu)
        print(c2.cobra.get_fbest())
        c2.p2.fe_thresh = 5e-02
        return c2

    def solve_G10(self, cobraSeed, feval=180, verbIter=10, conTol=1e-7):        # conTol=0 | 1e-7
        """ Test whether COP G10 has statistical equivalent results to the R side with squares=T, if we set on the
            Python side RBF.degree=2 (which is similar, but not the same).

            We test that the median of 15 final errors is < 5e-6, which is statistically equivalent to the R side
            (see ex_COP.R, function solve_G07, multi_gfnc)
        """
        print(f"Starting solve_G10({cobraSeed}) ...")
        G10 = GCOP("G10")
        idp = 9*10//2

        cobra = CobraInitializer(G10.x0, G10.fn, G10.name, G10.lower, G10.upper, G10.is_equ,
                                 solu=G10.solu,
                                 s_opts=SACoptions(verbose=verb, verboseIter=verbIter, feval=feval, cobraSeed=cobraSeed,
                                                   ID=IDoptions(initDesign="RAND_REP", initDesPoints=idp),
                                                   RBF=RBFoptions(degree=2),
                                                   ISA=ISAoptions(TGR=1e3),
                                                   SEQ=SEQoptions(finalEpsXiZero=True, conTol=conTol)))

        c2 = CobraPhaseII(cobra).start(gcop=G10)

        print(f"final err: {c2.p2.fin_err}")
        print(c2.p2.f_solu)
        print(c2.cobra.get_fbest())
        c2.p2.fe_thresh = 1e-9
        return c2

    def solve_G11(self, cobraSeed, feval=70, verbIter=10, conTol=0):          # conTol=0 | 1e-7
        """ Test whether COP G11 has statistical equivalent results to the R side with squares=T, if we set on the
            Python side RBF.degree=2 (which is similar, but not the same).

            We test that the median of 15 final errors is < 1e-13, which is statistically equivalent to the R side
            (see ex_COP.R, function solve_G11, multi_gfnc)
        """
        print(f"Starting solve_G11({cobraSeed}) ...")
        muFinal = 1e-4
        G11 = GCOP("G11", mu=muFinal)
        # G11.x0 = np.array([+np.sqrt(0.5-muFinal), 0.5])+0.1   # just to test, if the 2nd solution is found

        cobra = CobraInitializer(G11.x0, G11.fn, G11.name, G11.lower, G11.upper, G11.is_equ,
                                 solu=G11.solu,
                                 s_opts=SACoptions(verbose=verb, verboseIter=verbIter, feval=feval, cobraSeed=cobraSeed,
                                                   ID=IDoptions(initDesign="LHS", initDesPoints=6),
                                                   RBF=RBFoptions(degree=2),
                                                   EQU=EQUoptions(refinePrint=False, muFinal=muFinal, refineAlgo="COBYLA"),  # "L-BFGS-B COBYLA"
                                                   # COBYLA is slower, issues warnings, but is a bit more precise
                                                   SEQ=SEQoptions(finalEpsXiZero=True, conTol=conTol)))

        c2 = CobraPhaseII(cobra).start(gcop=G11)

        print(f"final err: {c2.p2.fin_err}")
        print(c2.p2.f_solu)
        print(c2.cobra.get_fbest())
        # c2.p2.fe_thresh = 1e-13     # this is for s_opts.SEQ.finalEpsXiZero=False
        c2.p2.fe_thresh = 1e-13       # this is for s_opts.SEQ.finalEpsXiZero=True
        return c2

    def solve_G12(self, cobraSeed, feval=140, verbIter=10, conTol=0):       # conTol=0 | 1e-7
        """ Test whether COP G12 has statistical equivalent results to the R side with squares=T, if we set on the
            Python side RBF.degree=2 (which is similar, but not the same).

            We test that the median of 15 final errors is < 1e-13, which is statistically equivalent to the R side
            (see ex_COP.R, function solve_G12, multi_gfnc)
        """
        print(f"Starting solve_G12({cobraSeed}) ...")
        G12 = GCOP("G12")

        cobra = CobraInitializer(G12.x0, G12.fn, G12.name, G12.lower, G12.upper, G12.is_equ,
                                 solu=G12.solu,
                                 s_opts=SACoptions(verbose=verb, verboseIter=verbIter, feval=feval, cobraSeed=cobraSeed,
                                                   ID=IDoptions(initDesign="LHS", initDesPoints=20),
                                                   RBF=RBFoptions(degree=2),
                                                   SEQ=SEQoptions(finalEpsXiZero=False, conTol=conTol)))

        c2 = CobraPhaseII(cobra).start(gcop=G12)

        print(f"final err: {c2.p2.fin_err}")
        print(c2.p2.f_solu)
        print(c2.cobra.get_fbest())
        c2.p2.fe_thresh = 1e-13
        return c2

    def solve_G13(self, cobraSeed, feval=500, verbIter=10, conTol=1e-7):
        """ Test whether COP G13 has statistical equivalent results to the R side with squares=T, if we set on the
            Python side RBF.degree=2 (which is similar, but not the same).

            We test that the median of 15 final errors is < 1e-13, which is statistically equivalent to the R side
            (see ex_COP.R, function solve_G11, multi_gfnc)
        """
        print(f"Starting solve_G13({cobraSeed}) ...")
        muFinal = 1e-4  # 1e-4, 1e-7
        G13 = GCOP("G13", mu=muFinal)
        dim = G13.dimension
        idp = (dim + 1) * (dim + 2) // 2

        equ = EQUoptions(muGrow=100, muDec=1.6, muFinal=muFinal,
                         refinePrint=False, refineAlgo="COBYLA")  # "L-BFGS-B COBYLA"
        cobra = CobraInitializer(G13.x0, G13.fn, G13.name, G13.lower, G13.upper, G13.is_equ,
                                 solu=G13.solu,
                                 s_opts=SACoptions(verbose=verb, verboseIter=verbIter, feval=feval, cobraSeed=cobraSeed,
                                                   ID=IDoptions(initDesign="LHS", initDesPoints=idp),
                                                   RBF=RBFoptions(degree=2, rho=2.5, rhoDec=2.0),  # , rhoGrow=100
                                                   EQU=equ,
                                                   SEQ=SEQoptions(finalEpsXiZero=True, conTol=conTol)))

        c2 = CobraPhaseII(cobra).start(gcop=G13)

        print(f"final err: {c2.p2.fin_err}")
        print(c2.p2.f_solu)
        print(c2.cobra.get_fbest())
        c2.p2.fe_thresh = 1e-8
        return c2

    def solve_G14(self, cobraSeed, feval=500, verbIter=50, conTol=0.0):  # , conTol=1e-7
        """ Test whether COP G14 has statistical equivalent results to the R side with squares=T, if we set on the
            Python side RBF.degree=2 (which is similar, but not the same).

            We test that the median of 15 final errors is < 1e-13, which is statistically equivalent to the R side
            (see ex_COP.R, function solve_G14, multi_gfnc)
        """
        print(f"Starting solve_G14({cobraSeed}) ...")
        muFinal = 1e-4  # 1e-4, 1e-7
        G14 = GCOP("G14", mu=muFinal)
        dim = G14.dimension
        idp = (dim + 1) * (dim + 2) // 2

        equ = EQUoptions(muGrow=100, muDec=1.6, muFinal=muFinal,
                         refinePrint=False, refineAlgo="L-BFGS-B")  # "L-BFGS-B COBYLA"
        cobra = CobraInitializer(G14.x0, G14.fn, G14.name, G14.lower, G14.upper, G14.is_equ,
                                 solu=G14.solu,
                                 s_opts=SACoptions(verbose=verb, verboseIter=verbIter, feval=feval, cobraSeed=cobraSeed,
                                                   ID=IDoptions(initDesign="LHS", initDesPoints=idp),
                                                   RBF=RBFoptions(degree=2, rho=2.5, rhoDec=2.0),  # , rhoGrow=100
                                                   EQU=equ,
                                                   SEQ=SEQoptions(finalEpsXiZero=True, conTol=conTol)))

        c2 = CobraPhaseII(cobra).start(gcop=G14)

        print(f"final err: {c2.p2.fin_err}")
        print(c2.p2.f_solu)
        print(c2.cobra.get_fbest())
        c2.p2.fe_thresh = 1e-1
        return c2

    def solve_G15(self, cobraSeed, feval=500, verbIter=50, conTol=0.0):       #, conTol=1e-7
        """ Test whether COP G15 has statistical equivalent results to the R side with squares=T, if we set on the
            Python side RBF.degree=2 (which is similar, but not the same).

            We test that the median of 15 final errors is < 1e-13, which is statistically equivalent to the R side
            (see ex_COP.R, function solve_G15, multi_gfnc)
        """
        print(f"Starting solve_G15({cobraSeed}) ...")
        muFinal = 1e-4  # 1e-4, 1e-7
        G15 = GCOP("G15", mu=muFinal)
        dim = G15.dimension
        idp = (dim + 1) * (dim + 2) // 2

        equ = EQUoptions(muGrow=100, muDec=1.6, muFinal=muFinal,
                         refinePrint=False, refineAlgo="L-BFGS-B")  # "L-BFGS-B COBYLA"
        cobra = CobraInitializer(G15.x0, G15.fn, G15.name, G15.lower, G15.upper, G15.is_equ,
                                 solu=G15.solu,
                                 s_opts=SACoptions(verbose=verb, verboseIter=verbIter, feval=feval, cobraSeed=cobraSeed,
                                                   ID=IDoptions(initDesign="LHS", initDesPoints=idp),
                                                   RBF=RBFoptions(degree=2),  # , rho=2.5, rhoDec=2.0, rhoGrow=100
                                                   EQU=equ,
                                                   SEQ=SEQoptions(finalEpsXiZero=True, conTol=conTol)))

        c2 = CobraPhaseII(cobra).start(gcop=G15)

        print(f"final err: {c2.p2.fin_err}")
        print(c2.p2.f_solu)
        print(c2.cobra.get_fbest())
        c2.p2.fe_thresh = 1e-1

        return c2

    def solve_G16(self, cobraSeed, feval=500, verbIter=50, conTol=0.0):       #, conTol=1e-7
        """ Test whether COP G16 has statistical equivalent results to the R side with squares=T, if we set on the
            Python side RBF.degree=2 (which is similar, but not the same).

            We test that the median of 15 final errors is < 1e-13, which is statistically equivalent to the R side
            (see ex_COP.R, function solve_G16, multi_gfnc)
        """
        print(f"Starting solve_G16({cobraSeed}) ...")
        G16 = GCOP("G16")
        dim = G16.dimension
        idp = (dim + 1) * (dim + 2) // 2

        cobra = CobraInitializer(G16.x0, G16.fn, G16.name, G16.lower, G16.upper, G16.is_equ,
                                 solu=G16.solu,
                                 s_opts=SACoptions(verbose=verb, verboseIter=verbIter, feval=feval, cobraSeed=cobraSeed,
                                                   ID=IDoptions(initDesign="LHS", initDesPoints=idp),
                                                   RBF=RBFoptions(degree=2),  # , rho=2.5, rhoDec=2.0, rhoGrow=100
                                                   SEQ=SEQoptions(finalEpsXiZero=True, conTol=conTol)))

        c2 = CobraPhaseII(cobra).start(gcop=G16)

        print(f"final err: {c2.p2.fin_err}")
        print(c2.p2.f_solu)
        print(c2.cobra.get_fbest())
        c2.p2.fe_thresh = 1e-1

        return c2

    def solve_G17(self, cobraSeed, feval=500, verbIter=50, conTol=0.0):    # conTol=1e-7
        """ Test whether COP G17 has statistical equivalent results to the R side with squares=T, if we set on the
            Python side RBF.degree=2 (which is similar, but not the same).

            We test that the median of 15 final errors is < 1e-13, which is statistically equivalent to the R side
            (see ex_COP.R, function solve_G17, multi_gfnc)
        """
        print(f"Starting solve_G17({cobraSeed}) ...")
        muFinal = 1e-4   # 1e-4 | 1e-7
        G17 = GCOP("G17", mu=muFinal)
        dim = G17.dimension
        idp = (dim + 1) * (dim + 2) // 2

        equ = EQUoptions(muGrow=100, muDec=1.6, muFinal=muFinal,
                         refinePrint=False, refineAlgo="L-BFGS-B")  # "L-BFGS-B" "COBYLA"
        cobra = CobraInitializer(G17.x0, G17.fn, G17.name, G17.lower, G17.upper, G17.is_equ,
                                 solu=G17.solu,
                                 s_opts=SACoptions(verbose=verb, verboseIter=verbIter, feval=feval, cobraSeed=cobraSeed,
                                                   ID=IDoptions(initDesign="LHS", initDesPoints=idp),
                                                   RBF=RBFoptions(degree=2, kernel="gaussian"),    #
                                                   EQU=equ,
                                                   ISA=ISAoptions(onlinePLOG=O_LOGIC.MIDPTS),
                                                   #RI=RIoptions(repairInfeas=True, eps2=0, q=3, repairMargin=np.inf, checkIt=False),
                                                   SEQ=SEQoptions(finalEpsXiZero=True, conTol=conTol)))

        c2 = CobraPhaseII(cobra).start(gcop=G17)

        print(f"final err: {c2.p2.fin_err}")
        print(c2.p2.f_solu)
        print(c2.cobra.get_fbest())
        c2.p2.fe_thresh = 1e-13
        analyze_solution(c2, cobra)
        return c2

    def solve_G18(self, cobraSeed, feval=500, verbIter=50, conTol=0.0):    # conTol=1e-7
        """ Test whether COP G18 has statistical equivalent results to the R side with squares=T, if we set on the
            Python side RBF.degree=2 (which is similar, but not the same).

            We test that the median of 15 final errors is < 1e-13, which is statistically equivalent to the R side
            (see ex_COP.R, function solve_G17, multi_gfnc)
        """
        print(f"Starting solve_G17({cobraSeed}) ...")
        G18 = GCOP("G18")
        dim = G18.dimension
        idp = (dim + 1) * (dim + 2) // 2

        cobra = CobraInitializer(G18.x0, G18.fn, G18.name, G18.lower, G18.upper, G18.is_equ,
                                 solu=G18.solu,
                                 s_opts=SACoptions(verbose=verb, verboseIter=verbIter, feval=feval, cobraSeed=cobraSeed,
                                                   ID=IDoptions(initDesign="LHS", initDesPoints=idp),
                                                   RBF=RBFoptions(degree=2, kernel="gaussian"),    #
                                                   ISA=ISAoptions(onlinePLOG=O_LOGIC.MIDPTS),
                                                   SEQ=SEQoptions(finalEpsXiZero=True, conTol=conTol)))

        c2 = CobraPhaseII(cobra).start(gcop=G18)

        print(f"final err: {c2.p2.fin_err}")
        print(c2.p2.f_solu)
        print(c2.cobra.get_fbest())
        c2.p2.fe_thresh = 1e-13
        analyze_solution(c2, cobra)
        return c2

    def solve_G19(self, cobraSeed, feval=500, verbIter=50, conTol=0.0):    # conTol=1e-7
        """ Test whether COP G19 has statistical equivalent results to the R side with squares=T, if we set on the
            Python side RBF.degree=2 (which is similar, but not the same).

            We test that the median of 15 final errors is < 1e-13, which is statistically equivalent to the R side
            (see ex_COP.R, function solve_G17, multi_gfnc)
        """
        print(f"Starting solve_G17({cobraSeed}) ...")
        G19 = GCOP("G19")
        dim = G19.dimension
        idp = (dim + 1) * (dim + 2) // 2

        cobra = CobraInitializer(G19.x0, G19.fn, G19.name, G19.lower, G19.upper, G19.is_equ,
                                 solu=G19.solu,
                                 s_opts=SACoptions(verbose=verb, verboseIter=verbIter, feval=feval, cobraSeed=cobraSeed,
                                                   ID=IDoptions(initDesign="LHS", initDesPoints=idp),
                                                   RBF=RBFoptions(degree=2, kernel="gaussian"),    #
                                                   ISA=ISAoptions(onlinePLOG=O_LOGIC.MIDPTS),
                                                   SEQ=SEQoptions(finalEpsXiZero=True, conTol=conTol)))

        c2 = CobraPhaseII(cobra).start(gcop=G19)

        print(f"final err: {c2.p2.fin_err}")
        print(c2.p2.f_solu)
        print(c2.cobra.get_fbest())
        c2.p2.fe_thresh = 1e-13
        analyze_solution(c2, cobra)
        return c2

    def solve_G21(self, cobraSeed, feval=500, verbIter=50, conTol=1e-4):
        """ Test whether COP G21 has statistical equivalent results to the R side with squares=T, if we set on the
            Python side RBF.degree=2 (which is similar, but not the same).

            We test that the median of 15 final errors is < 1e-13, which is statistically equivalent to the R side
            (see ex_COP.R, function solve_G17, multi_gfnc)
        """
        print(f"Starting solve_G21({cobraSeed}) ...")
        muFinal = 1e-4   # 1e-4 | 1e-7
        G21 = GCOP("G21", mu=muFinal)
        dim = G21.dimension
        idp = (dim + 1) * (dim + 2) // 2

        equ = EQUoptions(muGrow=100, muDec=1.6, muFinal=muFinal,        # 1e-7
                         refinePrint=False, refineAlgo="L-BFGS-B")  # "L-BFGS-B COBYLA"
        cobra = CobraInitializer(G21.x0, G21.fn, G21.name, G21.lower, G21.upper, G21.is_equ,
                                 solu=G21.solu,
                                 s_opts=SACoptions(verbose=verb, verboseIter=verbIter, feval=feval, cobraSeed=cobraSeed,
                                                   ID=IDoptions(initDesign="LHS", initDesPoints=idp),
                                                   RBF=RBFoptions(degree=2),
                                                   # ISA=ISAoptions(TGR=np.inf),
                                                   ISA=ISAoptions(onlinePLOG=O_LOGIC.MIDPTS, TGR=np.inf),   #
                                                   EQU=equ,
                                                   RI=RIoptions(repairInfeas=True, eps2=0, q=3, repairMargin=np.inf, checkIt=False), # new 2025/10/15
                                                   SEQ=SEQoptions(finalEpsXiZero=True, conTol=conTol,
                                                                  epsilonMax=0.0, epsilonInit=0.0, feMax=5000)))  #  , trueFuncForSurrogates=True
        c2 = CobraPhaseII(cobra).start(gcop=G21)

        print(f"final err: {c2.p2.fin_err}")
        print(c2.p2.f_solu)
        print(c2.cobra.get_fbest())
        c2.p2.fe_thresh = 1e-1
        # show_error_plot(cobra, G21, c2.get_muVec(), ylim=[1e-4,1e0])
        return c2

    def solve_G22(self, cobraSeed, feval=500, verbIter=50, conTol=1e-4):
        """ Test whether COP G22 has statistical equivalent results to the R side with squares=T, if we set on the
            Python side RBF.degree=2 (which is similar, but not the same).

            We test that the median of 15 final errors is < 1e-13, which is statistically equivalent to the R side
            (see ex_COP.R, function solve_G17, multi_gfnc)
        """
        print(f"Starting solve_G22({cobraSeed}) ...")
        muFinal = 1e-4   # 1e-4 | 1e-7
        G22 = GCOP("G22", mu=muFinal)
        dim = G22.dimension
        deg = 2
        idp = set_idp(dim, deg)

        equ = EQUoptions(muGrow=100, muDec=1.6, muFinal=muFinal,
                         refinePrint=False, refineAlgo="L-BFGS-B")  # "L-BFGS-B COBYLA"
        cobra = CobraInitializer(G22.x0, G22.fn, G22.name, G22.lower, G22.upper, G22.is_equ,
                                 solu=G22.solu,
                                 s_opts=SACoptions(verbose=verb, verboseIter=verbIter, feval=feval, cobraSeed=cobraSeed,
                                                   ID=IDoptions(initDesign="LHS", initDesPoints=idp),
                                                   RBF=RBFoptions(degree=deg),  # , interpolator="sacob"
                                                   ISA=ISAoptions(TGR=np.inf),
                                                   EQU=equ,
                                                   SEQ=SEQoptions(finalEpsXiZero=False, conTol=conTol)))  # , trueFuncForSurrogates=True
        c2 = CobraPhaseII(cobra).start(gcop=G22)
        # will also set various variables in c2.p2 via p2.fill()

        print(f"final err: {c2.p2.fin_err}")
        print(c2.p2.f_solu)
        print(c2.cobra.get_fbest())
        c2.p2.fe_thresh = 1e-1
        # show_error_plot(cobra, G22, c2.get_muVec(), ylim=[1e-4,1e0])
        return c2

    def solve_G23(self, cobraSeed, feval=500, verbIter=50, conTol=1e-4):
        """ Test whether COP G23 has statistical equivalent results to the R side with squares=T, if we set on the
            Python side RBF.degree=2 (which is similar, but not the same).

            We test that the median of 15 final errors is < 1e-13, which is statistically equivalent to the R side
            (see ex_COP.R, function solve_G17, multi_gfnc)
        """
        print(f"Starting solve_G23({cobraSeed}) ...")
        muFinal = 1e-7   # 1e-4 | 1e-7
        G23 = GCOP("G23", mu=muFinal)
        dim = G23.dimension
        deg = 2
        idp = set_idp(dim, deg)

        # debug only: Is a better solution found if we start right at the solution?
        # G23.x0 = G23.solu + 0.01

        equ = EQUoptions(muGrow=100, muDec=1.6, muFinal=muFinal, initType="useGrange",
                         refinePrint=False, refineAlgo="L-BFGS-B")  # "L-BFGS-B COBYLA"
        cobra = CobraInitializer(G23.x0, G23.fn, G23.name, G23.lower, G23.upper, G23.is_equ,
                                 solu=G23.solu,
                                 s_opts=SACoptions(verbose=verb, verboseIter=verbIter, feval=feval, cobraSeed=cobraSeed,
                                                   ID=IDoptions(initDesign="LHS", initDesPoints=idp),
                                                   RBF=RBFoptions(degree=deg),  # , interpolator="sacob"
                                                   # ISA=ISAoptions(TGR=np.inf),
                                                   ISA=ISAoptions(onlinePLOG=O_LOGIC.MIDPTS),   # run 2025/08/12   # , TGR=np.inf
                                                   EQU=equ,
                                                   RI=RIoptions(repairInfeas=True, eps2=0, q=3, repairMargin=np.inf, checkIt=False), # new 2025/10/15
                                                   SEQ=SEQoptions(finalEpsXiZero=True, conTol=conTol,
                                                                  epsilonMax=0.0, epsilonInit=0.0)))  # , feMax=1000 , trueFuncForSurrogates=True
        c2 = CobraPhaseII(cobra).start(gcop=G23)
        # will also set various variables in c2.p2 via p2.fill()

        print(f"final err: {c2.p2.fin_err}")
        print(c2.p2.f_solu)
        print(c2.cobra.get_fbest())
        c2.p2.fe_thresh = 1e-1
        # show_error_plot(cobra, G23, c2.get_muVec(), ylim=[1e-4,1e0])
        return c2

    def solve_G24(self, cobraSeed, feval=500, verbIter=50, conTol=1e-4):
        """ Test whether COP G24 has statistical equivalent results to the R side with squares=T, if we set on the
            Python side RBF.degree=2 (which is similar, but not the same).

            We test that the median of 15 final errors is < 1e-13, which is statistically equivalent to the R side
            (see ex_COP.R, function solve_G17, multi_gfnc)
        """
        print(f"Starting solve_G24({cobraSeed}) ...")
        G24 = GCOP("G24")
        dim = G24.dimension
        deg = 2
        idp = set_idp(dim, deg)

        cobra = CobraInitializer(G24.x0, G24.fn, G24.name, G24.lower, G24.upper, G24.is_equ,
                                 solu=G24.solu,
                                 s_opts=SACoptions(verbose=verb, verboseIter=verbIter, feval=feval, cobraSeed=cobraSeed,
                                                   ID=IDoptions(initDesign="LHS", initDesPoints=idp),
                                                   RBF=RBFoptions(degree=deg),  # , interpolator="sacob"
                                                   ISA=ISAoptions(TGR=np.inf),
                                                   SEQ=SEQoptions(finalEpsXiZero=False, conTol=conTol)))  # , trueFuncForSurrogates=True
        c2 = CobraPhaseII(cobra).start(gcop=G24)
        # will also set various variables in c2.p2 via p2.fill()

        print(f"final err: {c2.p2.fin_err}")
        print(c2.p2.f_solu)
        print(c2.cobra.get_fbest())
        c2.p2.fe_thresh = 1e-1
        # show_error_plot(cobra, G24, c2.get_muVec(), ylim=[1e-4,1e0])
        return c2

    def multi_gfnc(self, gfnc, gname: str, runs: int, cobraSeed: int):
        """ Perform multiple runs of COP ``gfnc`` with name ``gname``. The seed for run ``r in range(runs)`` is
            ``cobraSeed + r``.
        """
        start = time.perf_counter()
        fin_err_list = np.array([])
        c2 = None
        for run in range(runs):
            c2 = gfnc(cobraSeed + run, verbIter=100, conTol=1e-7)  #
            fin_err = c2.p2.fin_err
            fin_err_list = np.concatenate((fin_err_list, fin_err), axis=None)

        print(f"[{gname}] sorted {fin_err_list.size} final errors:")
        print(np.array(sorted(fin_err_list), dtype=float))  # to get rid of 'np.float64(...)'
        med_fin_err = np.median(fin_err_list)
        med_abs_fin_err = np.median(np.abs(fin_err_list))
        print(f"[{gname}] min: {np.min(fin_err_list):.6e},  max: {np.max(fin_err_list):.6e}")
        thresh = c2.p2.fe_thresh
        if med_fin_err <= thresh:
            print(f"[{gname}] median(final error) = {med_fin_err:.6e} is smaller than thresh = {thresh}")
            print(f"[{gname}] median(|final error|) = {med_abs_fin_err:.6e}")
        else:
            print(f"[{gname}] WARNING: median(final error) = {med_fin_err:.6e} is **NOT** smaller than thresh = {thresh}")
            print(f"[{gname}] median(|final error|) = {med_abs_fin_err:.6e}")
        print(f"[{gname}] ... finished ({(time.perf_counter() - start) / runs * 1000:.4f} msec per run, {runs} runs)")
        return c2

def analyze_solution(c2: CobraPhaseII, cobra: CobraInitializer):
    # compare xbest (as found by CobraPhaseII) with true solution. What is the maximum deviation in difference vector?
    print(f"cobra_xbest : {cobra.get_xbest()}")
    print(f"gcop_solu   : {cobra.solu}")
    print(f"d_xbest_solu: {cobra.get_xbest() - cobra.solu}")
    max_d_solu = np.max(np.abs(cobra.get_xbest() - cobra.solu))

    # compare true
    fn_xbest = cobra.sac_res['originalfn'](cobra.get_xbest())
    fn_solu = cobra.sac_res['originalfn'](cobra.solu)
    print(f"fn_xbest:   {fn_xbest}")
    print(f"fn_solu :   {fn_solu}")

    fsurr_xbest = c2.p2.fitnessSurrogate(c2.cobra.get_xbest_cobra())
    fsurr_solu = c2.p2.fitnessSurrogate(cobra.rw.forward(cobra.solu))
    print(f"fsurr_xbest:   {fsurr_xbest}")
    print(f"fsurr_solu :   {fsurr_solu}")

    csurr_xbest = c2.p2.constraintSurrogates(c2.cobra.get_xbest_cobra())
    csurr_solu = c2.p2.constraintSurrogates(cobra.rw.forward(cobra.solu))
    print(f"csurr_xbest:   {csurr_xbest}")
    print(f"csurr_solu :   {csurr_solu}")

    return fn_xbest, fn_solu, fsurr_xbest, fsurr_solu, csurr_xbest, csurr_solu, max_d_solu


def plot_func_surr(c2: CobraPhaseII, cobra: CobraInitializer, i, dim, d_range, png_file=None, ylim=None):
    """
    Plot a diagnostic visualization of optimization funcs and their surrogates:
    All plot lines are a cuts where input dimension ``dim`` is varied in range ``xbest[dim] +- d_range`` while all other
    input dimensions are left at their ``xbest``-values

    - red thick line: component ``i`` of true func ``fn``
    - blue dashed: component ``i`` of the surrogates (i=0: fitness, i>0: constraint i-1)
    - blue dotted: take for each input point ``x`` the nearest element from design matrix ``A`` and return the
      corresponding ``Fres|Gres`` value.

    Assertion (TODO): If the distance to the nearest row of design matrix is 0 or close to 0, then blue dot and red
    line should also coincide.

    :param c2:
    :param cobra:
    :param i:   the component of ``fn`` to visualize
    :param dim: the input dimension to vary
    :param d_range: the +- range over which to vary
    :param ylim: (optional) y-limits for the plot
    :return:
    """
    xbest = cobra.get_xbest()
    lower = cobra.sac_res['originalL']
    upper = cobra.sac_res['originalU']
    fn_func = lambda i,x: cobra.sac_res['originalfn'](x)[i]
    if i == 0:
        surr_func = lambda i,x: c2.p2.fitnessSurrogate(cobra.rw.forward(x))
    else:
        surr_func = lambda i, x: c2.p2.constraintSurrogates(cobra.rw.forward(x))[0][i-1]
    # darr = np.array([x for x in np.arange(xbest[dim]-d_range,xbest[dim]+d_range,2*d_range/100)])
    darr = np.array([x for x in np.arange(lower[dim], upper[dim], (upper[dim] - lower[dim]) / 100)])
    farr = darr * 0.0
    sarr = darr * 0.0
    sar2 = darr * 0.0
    marr = darr * 0.0
    x = xbest.copy()
    A = cobra.sac_res['A']
    F = cobra.sac_res['Fres']
    G = cobra.sac_res['Gres']
    FG = np.concatenate((F.reshape(F.shape[0], 1), G), axis=1)
    for k, d in enumerate(darr):
        x[dim] = d
        dist_x = distLine(cobra.rw.forward(x), A)
        min_ind = np.flatnonzero(dist_x == np.min(dist_x))[0]
        marr[k] = dist_x[min_ind]       # marr[k]: distance of k'th point to nearest point from A
        # x_close = A[min_ind, :]
        farr[k] = fn_func(i, x)
        sarr[k] = surr_func(i, x)
        sar2[k] = FG[min_ind, i]
    plt.close()
    plt.figure(figsize=(7, 6))  # (width,height) in inches
    plt.plot(darr, farr, 'r-', label='f', linewidth=2.2)  # thick red, to make it visible if otherwise blue sarr would overplot
    plt.plot(darr, sarr, 'b--', label='surr')   # '--' dashed line style
    plt.plot(darr, sar2, 'b:', label='s_A')     # ':'  dotted line style
    lower_k = max(fn_func(i,xbest)-100, np.min((farr,sarr,sar2)))
    upper_k = min(fn_func(i,xbest)+100, np.max((farr,sarr,sar2)))
    plt.plot([xbest[dim], xbest[dim]], [lower_k, upper_k], 'k-')
    plt.legend()
    plt.title(f"{cobra.sac_res['f_name']}, dim={cobra.get_xbest().size}", fontsize=20)
    plt.xlabel(f'x[{dim}] ', fontsize=16)
    plt.ylabel(f'fn[{i}]', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.subplot(111).set_yscale("linear")
    if ylim is not None:
        plt.subplot(111).set(ylim=ylim)
    if png_file is None:
        plt.show()
    else:
        plt.savefig(png_file)
    dummy = 0

if __name__ == '__main__':
    cop = ExamCOP()
    # exec("cop.solve_G06(42)")
    # cop.solve_G01(42)
    # cop.solve_G03(48, 7)
    # cop.solve_G04(53)
    # cop.solve_G05(42)
    # cop.solve_G06(42)
    # cop.solve_G07(42)
    # cop.solve_G11(42)
    # cop.solve_G12(42)
    # cop.solve_G13(62)
    # cop.solve_G14(62)
    # cop.solve_G15(62)
    # cop.solve_G17(54)
    # cop.solve_G21(63)
    # cop.solve_G22(55, conTol=0.0, verbIter=10)
    cc2 = cop.multi_gfnc(cop.solve_G05, "G05", 5, 49)
    # cc2 = cop.multi_gfnc(cop.solve_G04, "G04", 15, 42)
    # cc2 = cop.multi_gfnc(cop.solve_G15, "G15", 10, 48)
    # cc2 = cop.multi_gfnc(cop.solve_G17, "G17", 10, 61)
    # cc2 = cop.multi_gfnc(cop.solve_G14, "G14", 6, 54)
    # cc2 = cop.multi_gfnc(cop.solve_G01, "G01", 6, 54)
    # cc2 = cop.multi_gfnc(cop.solve_G09, "G09", 10, 54)
    # cc2 = cop.multi_gfnc(cop.solve_G02, "G02", 10, 54)
    # cop.solve_G03(57, dimension=8, feval=500, verbIter=20)

