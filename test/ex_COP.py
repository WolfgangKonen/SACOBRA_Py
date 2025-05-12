import time
import unittest
import numpy as np
from cobraInit import CobraInitializer
from gCOP import GCOP
from cobraPhaseII import CobraPhaseII
from opt.equOptions import EQUoptions
from opt.sacOptions import SACoptions
from opt.idOptions import IDoptions
from opt.rbfOptions import RBFoptions
from opt.seqOptions import SEQoptions

verb = 1


class ExamCOP:
    """
        Example COPs from the G function benchmark. Test for statistical equivalence to the R side (ex_COP.R)

        - G06 is a COP with two circular inequality constraints that form a very narrow feasible region. d=2.
        - G05 is a COP with 2 inequality and 3 equality constraints and d=4.
        - G04 is a COP with 6 inequality constraints and d=5.
        - G03 is a COP with 1 equality constraint (sphere) and steerable dimension d.
    """
    def solve_G03(self, cobraSeed, dimension=7):
        """ Test whether COP G03 has statistical similar results to the R side with squares=T, if we set on the
            Python side RBF.degree=2 (which is similar, but not the same).

            We test that the median of 15 final errors is < 1e-9, which is statistically better than the R side
            (see ex_COP.R)
        """
        print(f"Starting solve_G03({cobraSeed}) ...")
        G03 = GCOP("G03", dimension)

        equ = EQUoptions(dec=1.6, equEpsFinal=1e-7, refinePrint=False, refineAlgo="L-BFGS-B")  # "L-BFGS-B COBYLA"
        # x0 = G03.x0             # None --> a random x0 will be set
        x0 = np.arange(dimension)/dimension    # fixed x0
        cobra = CobraInitializer(x0, G03.fn, G03.name, G03.lower, G03.upper,
                                 is_equ=G03.is_equ,
                                 s_opts=SACoptions(verbose=verb, verboseIter=10, feval=150, cobraSeed=cobraSeed,
                                                   finalEpsXiZero=True,
                                                   ID=IDoptions(initDesign="RAND_REP", rescale=True),
                                                   RBF=RBFoptions(degree=1),
                                                   EQU=equ,
                                                   SEQ=SEQoptions(conTol=0)))     # conTol=1e-7
        print(f"idp = {cobra.sac_opts.ID.initDesPoints}")
        c2 = CobraPhaseII(cobra)
        c2.start()

        fin_err = np.array(cobra.get_fbest() - G03.fn(G03.solu)[0])
        print(f"final err: {fin_err}")
        c2.p2.fin_err = fin_err
        c2.p2.fe_thresh = 1e-9
        return c2

    def solve_G04(self, cobraSeed):
        """ Test whether COP G04 has statistical similar results to the R side with squares=T, if we set on the
            Python side RBF.degree=2 (which is similar, but not the same).

            We test that the median of 15 final errors is < 1e-9 (actually 6e-11 for rescale=False and 2e-10 for
            rescale=True), which is statistically better than the R side (see ex_COP.R)
        """
        print(f"Starting solve_G04({cobraSeed}) ...")
        G04 = GCOP("G04")

        cobra = CobraInitializer(G04.x0, G04.fn, G04.name, G04.lower, G04.upper,
                                 is_equ=G04.is_equ,
                                 s_opts=SACoptions(verbose=verb, verboseIter=10, feval=170, cobraSeed=cobraSeed,
                                                   finalEpsXiZero=True,
                                                   ID=IDoptions(initDesign="RAND_REP", rescale=False),
                                                   RBF=RBFoptions(degree=2),
                                                   EQU=EQUoptions(dec=1.6, equEpsFinal=1e-7, refinePrint=False),
                                                   SEQ=SEQoptions(conTol=0)))     # conTol=1e-7
        c2 = CobraPhaseII(cobra)
        c2.start()

        fin_err = np.array(cobra.get_fbest() - G04.fn(G04.solu)[0])
        print(f"final err: {fin_err}")
        c2.p2.fin_err = fin_err
        c2.p2.fe_thresh = 1e-9
        return c2

    def solve_G05(self, cobraSeed):
        """ Test whether COP G05 has statistical similar results to the R side with squares=T, if we set on the
            Python side RBF.degree=2 (which is similar, but not the same).

            We test that the median of 15 final errors is < 5e-6, which is statistically similar to the R side
            (see ex_COP.R)
        """
        G05 = GCOP("G05")
        idp = 15   # =(d+1)(d+2)/2, the minimum for RBF.kernel="cubic", RBF.degree=2 and d=4

        cobra = CobraInitializer(G05.x0, G05.fn, G05.name, G05.lower, G05.upper,
                                 is_equ=G05.is_equ,
                                 s_opts=SACoptions(verbose=verb, feval=170, cobraSeed=cobraSeed,
                                                   finalEpsXiZero=True,
                                                   ID=IDoptions(initDesign="RAND_REP", initDesPoints=idp),
                                                   RBF=RBFoptions(degree=2),
                                                   EQU=EQUoptions(dec=1.6, equEpsFinal=1e-7, refinePrint=False),
                                                   SEQ=SEQoptions(conTol=0)))     # conTol=1e-7
        c2 = CobraPhaseII(cobra)
        c2.start()

        fin_err = np.array(cobra.get_fbest() - G05.fn(G05.solu)[0])
        print(f"final err: {fin_err}")
        c2.p2.fin_err = fin_err
        c2.p2.fe_thresh = 5e-6
        return c2

    def solve_G06(self, cobraSeed):
        """ Test whether COP G06 has statistical equivalent results to the R side with squares=T, if we set on the
            Python side RBF.degree=2 (which is similar, but not the same).

            We test that the median of 15 final errors is < 5e-6, which is statistically equivalent to the R side
            (see ex_COP.R, function multi_G06)
        """
        G06 = GCOP("G06")

        cobra = CobraInitializer(G06.x0, G06.fn, G06.name, G06.lower, G06.upper,
                                 s_opts=SACoptions(verbose=verb, feval=40, cobraSeed=cobraSeed,
                                                   finalEpsXiZero=True,
                                                   ID=IDoptions(initDesign="RAND_REP", initDesPoints=6),
                                                   RBF=RBFoptions(degree=2),
                                                   SEQ=SEQoptions(conTol=1e-7)))     # trueFuncForSurrogates=True

        c2 = CobraPhaseII(cobra)
        c2.start()

        fin_err = np.array(cobra.get_fbest() - G06.fn(G06.solu)[0])
        c2.p2.fin_err = fin_err
        print(f"final err: {fin_err}")
        # c2.p2.fe_thresh = 5e-6    # this is for s_opts.finalEpsXiZero=False
        c2.p2.fe_thresh = 5e-8      # this is for s_opts.finalEpsXiZero=True and s_opts.SEQ.conTol=1e-7
        return c2

    def multi_gfnc(self, gfnc, gname, runs, cobraSeed):
        """ Test whether COP ``gfnc`` with name ``gname`` has statistical equivalent results to the R side with
            squares=T, if we set on the Python side RBF.degree=2 (which is similar, but not the same).
        """
        start = time.perf_counter()
        fin_err_list = np.array([])
        for run in range(runs):
            c2 = gfnc(cobraSeed + run)
            # cobra = c2.get_cobra()
            fin_err = c2.p2.fin_err
            fin_err_list = np.concatenate((fin_err_list, fin_err), axis=None)
            # print(f"final err: {fin_err}")

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
    # cop.solve_G03(48, 7)
    # cop.solve_G04(53)
    # cop.solve_G05(42)
    c2 = cop.multi_gfnc(cop.solve_G04, "G04", 15, 42)
    # c2 = cop.multi_gfnc(cop.solve_G03, "G03", 6, 48)
