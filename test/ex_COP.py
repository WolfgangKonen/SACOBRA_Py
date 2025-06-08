import time
import numpy as np
import pandas as pd

from cobraInit import CobraInitializer
from gCOP import GCOP
from cobraPhaseII import CobraPhaseII
from opt.equOptions import EQUoptions
from opt.isaOptions import ISAoptions
from opt.sacOptions import SACoptions
from opt.idOptions import IDoptions
from opt.rbfOptions import RBFoptions
from opt.seqOptions import SEQoptions
from show_error_plot import show_error_plot

verb = 1


class ExamCOP:
    """
        Example COPs from the G function benchmark. Test for statistical equivalence to the R side (ex_COP.R)

        - G01 is a COP with 9 linear inequality constraints and d=13
        - G03 is a COP with 1 equality constraint (sphere) and steerable dimension d.
        - G04 is a COP with 6 inequality constraints and d=5.
        - G05 is a COP with 2 inequality and 3 equality constraints. d=4.
        - G06 is a COP with two circular inequality constraints that form a very narrow feasible region. d=2.
        - G07 is a COP with 8 inequality constraints. d=10.
        - G11 is a COP with 1 equality constraint. d=2.
        - G13 is a COP with 3 equality constraints. d=5.
        - G14 is a COP with 3 equality constraints. d=10.
        - G15 is a COP with 5 equality constraints. d=3.
        - G17 is a COP with 4 equality constraints. d=6.
        - G21 is a COP with 5 equality constraints. d=7.

        In summary, there are 8 COPs (G03, G05, G11, G13, G14, G15, G17, G21) containing equality constraints.
    """

    def solve_G01(self, cobraSeed, feval=170, verbIter=10):
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
                                                   ID=IDoptions(initDesign="RAND_REP", initDesPoints=idp),
                                                   RBF=RBFoptions(degree=2),
                                                   SEQ=SEQoptions(finalEpsXiZero=False, conTol=0)))     # conTol=1e-7
        c2 = CobraPhaseII(cobra).start()

        fin_err = np.array(cobra.get_fbest() - G01.fbest)
        print(f"final err: {fin_err}")
        c2.p2.fin_err = fin_err
        c2.p2.fe_thresh = 5e-6      # same accuracy 1.1e-6 for s_opts.SEQ.finalEpsXiZero=True or False
        c2.p2.dim = G01.dimension
        return c2

    def solve_G03(self, cobraSeed, dimension=7, feval=150, verbIter=10):
        """ Test whether COP G03 has statistical similar results to the R side with squares=T, if we set on the
            Python side RBF.degree=2 (which is similar, but not the same).

            We test that the median of 15 final errors is < 1e-9, which is statistically better than the R side
            (see ex_COP.R)
        """
        print(f"Starting solve_G03({cobraSeed}, ...) ...")
        G03 = GCOP("G03", dimension)

        equ = EQUoptions(muDec=1.6, muFinal=1e-7, refinePrint=False, refineAlgo="L-BFGS-B")  # "L-BFGS-B COBYLA"
        # x0 = G03.x0             # None --> a random x0 will be set
        x0 = np.arange(dimension)/dimension    # fixed x0
        cobra = CobraInitializer(x0, G03.fn, G03.name, G03.lower, G03.upper, G03.is_equ,
                                 s_opts=SACoptions(verbose=verb, verboseIter=verbIter, feval=feval, cobraSeed=cobraSeed,
                                                   ID=IDoptions(initDesign="RAND_REP", rescale=True),
                                                   RBF=RBFoptions(degree=1),
                                                   EQU=equ,
                                                   SEQ=SEQoptions(finalEpsXiZero=True, conTol=0)))     # conTol=1e-7
        print(f"idp = {cobra.sac_opts.ID.initDesPoints}")
        c2 = CobraPhaseII(cobra).start()

        fin_err = np.array(cobra.get_fbest() - G03.fbest)
        print(f"final err: {fin_err}")
        c2.p2.fin_err = fin_err
        c2.p2.fe_thresh = 1e-9
        c2.p2.dim = G03.dimension
        return c2

    def solve_G04(self, cobraSeed, feval=170, verbIter=10):
        """ Test whether COP G04 has statistical similar results to the R side with squares=T, if we set on the
            Python side RBF.degree=2 (which is similar, but not the same).

            We test that the median of 15 final errors is < 1e-9 (actually 6e-11 for rescale=False and 2e-10 for
            rescale=True), which is statistically better than the R side (see ex_COP.R)
        """
        print(f"Starting solve_G04({cobraSeed}) ...")
        G04 = GCOP("G04")

        cobra = CobraInitializer(G04.x0, G04.fn, G04.name, G04.lower, G04.upper, G04.is_equ,
                                 s_opts=SACoptions(verbose=verb, verboseIter=verbIter, feval=feval, cobraSeed=cobraSeed,
                                                   ID=IDoptions(initDesign="LHS", rescale=False),
                                                   RBF=RBFoptions(degree=2),
                                                   EQU=EQUoptions(muDec=1.6, muFinal=1e-7, refinePrint=False),
                                                   SEQ=SEQoptions(finalEpsXiZero=True, conTol=0)))  # conTol=0 | 1e-7
        c2 = CobraPhaseII(cobra).start()

        fin_err = np.array(cobra.get_fbest() - G04.fbest)
        print(f"final err: {fin_err}")
        c2.p2.fin_err = fin_err
        c2.p2.fe_thresh = 1e-9
        c2.p2.dim = G04.dimension
        return c2

    def solve_G05(self, cobraSeed, feval=170, verbIter=10):
        """ Test whether COP G05 has statistical similar results to the R side with squares=T, if we set on the
            Python side RBF.degree=2 (which is similar, but not the same).

            We test that the median of 15 final errors is < 5e-6, which is statistically similar to the R side
            (see ex_COP.R)
        """
        print(f"Starting solve_G05({cobraSeed}) ...")
        G05 = GCOP("G05")
        idp = 15   # =(d+1)(d+2)/2, the minimum for RBF.kernel="cubic", RBF.degree=2 and d=4

        cobra = CobraInitializer(G05.x0, G05.fn, G05.name, G05.lower, G05.upper, G05.is_equ,
                                 solu=G05.solu,
                                 s_opts=SACoptions(verbose=verb, verboseIter=verbIter, feval=feval, cobraSeed=cobraSeed,
                                                   ID=IDoptions(initDesign="LHS", initDesPoints=idp),
                                                   RBF=RBFoptions(degree=2),
                                                   EQU=EQUoptions(muDec=1.6, muFinal=1e-7, refinePrint=False,
                                                                  refineAlgo="COBYLA"),  # "L-BFGS-B COBYLA"
                                                   SEQ=SEQoptions(finalEpsXiZero=True, conTol=0)))   # conTol=0 | 1e-7
        c2 = CobraPhaseII(cobra).start()

        fin_err = np.array(cobra.get_fbest() - G05.fbest)
        print(f"final err: {fin_err}")
        c2.p2.fin_err = fin_err
        c2.p2.fe_thresh = 5e-6
        c2.p2.dim = G05.dimension
        return c2

    def solve_G06(self, cobraSeed, feval=40, verbIter=10):
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
                                                   SEQ=SEQoptions(finalEpsXiZero=True, conTol=0)))   # conTol=0 | 1e-7

        c2 = CobraPhaseII(cobra)
        c2.start()

        # show_error_plot(cobra, G06)

        fin_err = np.array(cobra.get_fbest() - G06.fbest)
        c2.p2.fin_err = fin_err
        print(f"final err: {fin_err}")
        # c2.p2.fe_thresh = 5e-6    # this is for s_opts.SEQ.finalEpsXiZero=False
        c2.p2.fe_thresh = 5e-8      # this is for s_opts.SEQ.finalEpsXiZero=True and s_opts.SEQ.conTol=1e-7
        c2.p2.dim = G06.dimension
        return c2

    def solve_G07(self, cobraSeed, feval=180, verbIter=10):
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
                                                   SEQ=SEQoptions(finalEpsXiZero=True, conTol=0)))   # conTol=0 | 1e-7

        c2 = CobraPhaseII(cobra).start()

        fin_err = np.array(cobra.get_fbest() - G07.fbest)
        c2.p2.fin_err = fin_err
        print(f"final err: {fin_err}")
        c2.p2.fe_thresh = 1e-9
        c2.p2.dim = G07.dimension
        return c2

    def solve_G11(self, cobraSeed, feval=70, verbIter=10):
        """ Test whether COP G11 has statistical equivalent results to the R side with squares=T, if we set on the
            Python side RBF.degree=2 (which is similar, but not the same).

            We test that the median of 15 final errors is < 1e-13, which is statistically equivalent to the R side
            (see ex_COP.R, function solve_G11, multi_gfnc)
        """
        print(f"Starting solve_G11({cobraSeed}) ...")
        G11 = GCOP("G11")

        cobra = CobraInitializer(G11.x0, G11.fn, G11.name, G11.lower, G11.upper, G11.is_equ,
                                 solu=G11.solu,
                                 s_opts=SACoptions(verbose=verb, verboseIter=verbIter, feval=feval, cobraSeed=cobraSeed,
                                                   ID=IDoptions(initDesign="LHS", initDesPoints=6),
                                                   RBF=RBFoptions(degree=2),
                                                   EQU=EQUoptions(refinePrint=False, refineAlgo="COBYLA"),  # "L-BFGS-B COBYLA"
                                                   # COBYLA is slower, issues warnings, but is a bit more precise
                                                   SEQ=SEQoptions(finalEpsXiZero=True, conTol=0)))   # conTol=0 | 1e-7

        c2 = CobraPhaseII(cobra).start()

        fin_err = np.array(cobra.get_fbest() - G11.fbest)
        c2.p2.fin_err = fin_err
        print(f"final err: {fin_err}")
        # c2.p2.fe_thresh = 1e-13     # this is for s_opts.SEQ.finalEpsXiZero=False
        c2.p2.fe_thresh = 1e-13       # this is for s_opts.SEQ.finalEpsXiZero=True
        c2.p2.dim = G11.dimension
        return c2

    def solve_G12(self, cobraSeed, feval=140, verbIter=10):
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
                                                   SEQ=SEQoptions(finalEpsXiZero=False, conTol=0)))   # conTol=0 | 1e-7

        c2 = CobraPhaseII(cobra).start()

        fin_err = np.array(cobra.get_fbest() - G12.fbest)
        c2.p2.fin_err = fin_err
        print(f"final err: {fin_err}")
        c2.p2.fe_thresh = 1e-13
        c2.p2.dim = G12.dimension
        return c2

    def solve_G13(self, cobraSeed, feval=500, verbIter=10):
        """ Test whether COP G13 has statistical equivalent results to the R side with squares=T, if we set on the
            Python side RBF.degree=2 (which is similar, but not the same).

            We test that the median of 15 final errors is < 1e-13, which is statistically equivalent to the R side
            (see ex_COP.R, function solve_G11, multi_gfnc)
        """
        print(f"Starting solve_G13({cobraSeed}) ...")
        G13 = GCOP("G13")
        idp = 6 * 7 // 2

        equ = EQUoptions(muGrow=100, muDec=1.6, muFinal=1e-7,
                         refinePrint=False, refineAlgo="COBYLA")  # "L-BFGS-B COBYLA"
        cobra = CobraInitializer(G13.x0, G13.fn, G13.name, G13.lower, G13.upper, G13.is_equ,
                                 solu=G13.solu,
                                 s_opts=SACoptions(verbose=verb, verboseIter=verbIter, feval=feval, cobraSeed=cobraSeed,
                                                   ID=IDoptions(initDesign="LHS", initDesPoints=idp),
                                                   RBF=RBFoptions(degree=2, rho=2.5, rhoDec=2.0),  # , rhoGrow=100
                                                   EQU=equ,
                                                   SEQ=SEQoptions(finalEpsXiZero=True, conTol=1e-7)))

        c2 = CobraPhaseII(cobra).start()

        fin_err = np.array(cobra.get_fbest() - G13.fbest)
        c2.p2.fin_err = fin_err
        print(f"final err: {fin_err}")
        c2.p2.fe_thresh = 1e-8
        c2.p2.dim = G13.dimension
        return c2

    def solve_G14(self, cobraSeed, feval=500, verbIter=10):
        """ Test whether COP G14 has statistical equivalent results to the R side with squares=T, if we set on the
            Python side RBF.degree=2 (which is similar, but not the same).

            We test that the median of 15 final errors is < 1e-13, which is statistically equivalent to the R side
            (see ex_COP.R, function solve_G11, multi_gfnc)
        """
        print(f"Starting solve_G14({cobraSeed}) ...")
        G14 = GCOP("G14")
        idp = 11 * 12 // 2

        equ = EQUoptions(muGrow=100, muDec=1.6, muFinal=1e-7,
                         refinePrint=False, refineAlgo="L-BFGS-B")  # "L-BFGS-B COBYLA"
        cobra = CobraInitializer(G14.x0, G14.fn, G14.name, G14.lower, G14.upper, G14.is_equ,
                                 solu=G14.solu,
                                 s_opts=SACoptions(verbose=verb, verboseIter=verbIter, feval=feval, cobraSeed=cobraSeed,
                                                   ID=IDoptions(initDesign="LHS", initDesPoints=idp),
                                                   RBF=RBFoptions(degree=2, rho=2.5, rhoDec=2.0),  # , rhoGrow=100
                                                   EQU=equ,
                                                   SEQ=SEQoptions(finalEpsXiZero=True, conTol=1e-7)))     # 1e-7

        c2 = CobraPhaseII(cobra).start()

        fin_err = np.array(cobra.get_fbest() - G14.fbest)
        c2.p2.fin_err = fin_err
        print(f"final err: {fin_err}")
        c2.p2.fe_thresh = 1e-1
        c2.p2.dim = G14.dimension
        return c2

    def solve_G15(self, cobraSeed):
        """ Test whether COP G15 has statistical equivalent results to the R side with squares=T, if we set on the
            Python side RBF.degree=2 (which is similar, but not the same).

            We test that the median of 15 final errors is < 1e-13, which is statistically equivalent to the R side
            (see ex_COP.R, function solve_G15, multi_gfnc)
        """
        print(f"Starting solve_G15({cobraSeed}) ...")
        G15 = GCOP("G15")
        idp = 11 * 12 // 2

        equ = EQUoptions(muGrow=100, muDec=1.6, muFinal=1e-7,
                         refinePrint=False, refineAlgo="L-BFGS-B")  # "L-BFGS-B COBYLA"
        cobra = CobraInitializer(G15.x0, G15.fn, G15.name, G15.lower, G15.upper, G15.is_equ,
                                 solu=G15.solu,
                                 s_opts=SACoptions(verbose=verb, feval=300, cobraSeed=cobraSeed,
                                                   ID=IDoptions(initDesign="LHS", initDesPoints=idp),
                                                   RBF=RBFoptions(degree=2),  # , rho=2.5, rhoDec=2.0, rhoGrow=100
                                                   EQU=equ,
                                                   SEQ=SEQoptions(finalEpsXiZero=True, conTol=1e-7)))     # 1e-7

        c2 = CobraPhaseII(cobra).start()

        fin_err = np.array(cobra.get_fbest() - G15.fbest)
        c2.p2.fin_err = fin_err
        print(f"final err: {fin_err}")
        c2.p2.fe_thresh = 1e-1
        return c2

    def solve_G17(self, cobraSeed):
        """ Test whether COP G17 has statistical equivalent results to the R side with squares=T, if we set on the
            Python side RBF.degree=2 (which is similar, but not the same).

            We test that the median of 15 final errors is < 1e-13, which is statistically equivalent to the R side
            (see ex_COP.R, function solve_G17, multi_gfnc)
        """
        print(f"Starting solve_G17({cobraSeed}) ...")
        G17 = GCOP("G17")
        idp = 7 * 8//2

        equ = EQUoptions(muGrow=100, muDec=1.6, muFinal=1e-7,
                         refinePrint=False, refineAlgo="COBYLA")  # "L-BFGS-B COBYLA"
        cobra = CobraInitializer(G17.x0, G17.fn, G17.name, G17.lower, G17.upper, G17.is_equ,
                                 solu=G17.solu,
                                 s_opts=SACoptions(verbose=verb, feval=500, cobraSeed=cobraSeed,
                                                   ID=IDoptions(initDesign="LHS", initDesPoints=idp),
                                                   RBF=RBFoptions(degree=2),
                                                   EQU=equ,
                                                   SEQ=SEQoptions(finalEpsXiZero=False, conTol=1e-7)))

        c2 = CobraPhaseII(cobra).start()

        fin_err = np.array(cobra.get_fbest() - G17.fbest)
        c2.p2.fin_err = fin_err
        print(f"final err: {fin_err}")
        c2.p2.fe_thresh = 1e-13
        return c2

    def solve_G21(self, cobraSeed):
        """ Test whether COP G21 has statistical equivalent results to the R side with squares=T, if we set on the
            Python side RBF.degree=2 (which is similar, but not the same).

            We test that the median of 15 final errors is < 1e-13, which is statistically equivalent to the R side
            (see ex_COP.R, function solve_G17, multi_gfnc)
        """
        print(f"Starting solve_G21({cobraSeed}) ...")
        G21 = GCOP("G21")
        idp = 8 * 9//2

        equ = EQUoptions(muGrow=100, muDec=1.6, muFinal=1e-4,
                         refinePrint=False, refineAlgo="L-BFGS-B")  # "L-BFGS-B COBYLA"
        cobra = CobraInitializer(G21.x0, G21.fn, G21.name, G21.lower, G21.upper, G21.is_equ,
                                 solu=G21.solu,
                                 s_opts=SACoptions(verbose=verb, feval=500, cobraSeed=cobraSeed,
                                                   ID=IDoptions(initDesign="LHS", initDesPoints=idp),
                                                   RBF=RBFoptions(degree=2),
                                                   # ISA=ISAoptions(TGR=np.inf),
                                                   EQU=equ,
                                                   SEQ=SEQoptions(finalEpsXiZero=False, conTol=1e-4)))
        c2 = CobraPhaseII(cobra).start()

        fin_err = np.array(cobra.get_fbest() - G21.fbest)
        c2.p2.fin_err = fin_err
        print(f"final err: {fin_err}")
        c2.p2.fe_thresh = 1e-1
        # show_error_plot(cobra, G21, ylim=[1e-4,1e0])
        return c2

    def multi_gfnc(self, gfnc, gname: str, runs: int, cobraSeed: int):
        """ Perform multiple runs of COP ``gfnc`` with name ``gname``. The seed for run ``r in range(runs)`` is
            ``cobraSeed + r``.
        """
        start = time.perf_counter()
        fin_err_list = np.array([])
        c2 = None
        for run in range(runs):
            c2 = gfnc(cobraSeed + run)
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

if __name__ == '__main__':
    cop = ExamCOP()
    # exec("cop.solve_G06(42)")
    # cop.solve_G01(42)
    # cop.solve_G03(48, 7)
    # cop.solve_G04(53)
    # cop.solve_G05(42)
    # cop.solve_G06(42)
    # cop.solve_G07(42)
    cop.solve_G11(42)
    # cop.solve_G12(42)
    # cop.solve_G13(62)
    # cop.solve_G14(62)
    # cop.solve_G17(62)
    # cop.solve_G21(63)
    # cc2 = cop.multi_gfnc(cop.solve_G01, "G01", 10, 48)
    # cc2 = cop.multi_gfnc(cop.solve_G04, "G04", 15, 42)
    # cc2 = cop.multi_gfnc(cop.solve_G15, "G15", 10, 48)
    # cc2 = cop.multi_gfnc(cop.solve_G17, "G17", 10, 54)
    # cc2 = cop.multi_gfnc(cop.solve_G14, "G14", 6, 54)
    # cc2 = cop.multi_gfnc(cop.solve_G01, "G01", 6, 54)
    # cc2 = cop.multi_gfnc(cop.solve_G21, "G21", 10, 54)
    cop.solve_G04(42, feval=170, verbIter=10)

