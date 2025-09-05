import os
import time
import numpy as np
import pandas as pd

from cobraInit import CobraInitializer
from ex_COP import ExamCOP
from gCOP import GCOP, show_error_plot
from cobraPhaseII import CobraPhaseII
from opt.equOptions import EQUoptions
from opt.isaOptions import ISAoptions, O_LOGIC
from opt.sacOptions import SACoptions
from opt.idOptions import IDoptions
from opt.rbfOptions import RBFoptions
from opt.seqOptions import SEQoptions

verb = 1


class OneS:
    def one_s(self, gname: str, dim: int, cobraSeed: int, feval=300, conTol=0.0):
        """ One SACOBRA configuration for all G-problems.

            Run this configuration on COP ``gname`` with given seed, using ``feval`` and ``conTol`` as specified
            or as given by the defaults.

        :param gname:       name of G-problem
        :param cobraSeed:   seed
        :param feval:       real function evaluations
        :param conTol:      constraint tolerance, common values are 0 or 1e-7
        :return:    ``c2``, the resulting object after running ``CobraPhaseII.start()``
        """
        print(f"Starting one_s({gname}, dim={dim}, {cobraSeed}) ...")
        if gname in {"G02", "G03"}:
            gcop = GCOP(gname, dimension=dim)
        else:
            gcop = GCOP(gname)

        dim = gcop.dimension
        idp = (dim + 1) * (dim + 2) // 2
        if feval == 0: feval = idp+2

        equ = EQUoptions(muGrow=100, muDec=1.6, muFinal=1e-7,
                         refinePrint=False, refineAlgo="L-BFGS-B")  # "L-BFGS-B COBYLA"
        cobra = CobraInitializer(gcop.x0, gcop.fn, gcop.name, gcop.lower, gcop.upper, gcop.is_equ,
                                 solu=gcop.solu,
                                 s_opts=SACoptions(verbose=verb, verboseIter=100, feval=feval, cobraSeed=cobraSeed,
                                                   ID=IDoptions(initDesign="LHS", initDesPoints=idp),
                                                   RBF=RBFoptions(degree=2),   # for default interpolator="scipy" + "cubic"
                                                   # RBF=RBFoptions(degree=1.5, interpolator="sacobra"),  # test only, "cubic"
                                                   # RBF=RBFoptions(kernel="gaussian", degree=2),   # alternative "gaussian"
                                                   # ISA=ISAoptions(onlinePLOG=O_LOGIC.NONE),   # the default (before 2025/08/01)
                                                   ISA=ISAoptions(onlinePLOG=O_LOGIC.MIDPTS), # run 2025/08/12
                                                   # ISA=ISAoptions(onlinePLOG=O_LOGIC.XNEW),     # run 2025/08/13
                                                   EQU=equ,
                                                   SEQ=SEQoptions(finalEpsXiZero=True, conTol=conTol)))  # , trueFuncForSurrogates=True
        if feval > idp: c2 = CobraPhaseII(cobra).start()

        fin_err = np.array(cobra.get_fbest() - gcop.fbest)
        if fin_err < 1e-7:
            dummy = 0
        c2.p2.fin_err = fin_err
        print(f"final err: {fin_err}")
        c2.p2.fe_thresh = 1e-1
        c2.p2.dim = dim
        c2.p2.conTol = conTol
        # show_error_plot(cobra, gcop,)  #  ylim=[1e-4,1e0]
        # print(gcop.fn(gcop.solu))
        print(gcop.fbest)
        print(c2.cobra.get_fbest())
        return c2

    def one_s_multi_g_r(self, gnames: list, dims: list, runs: int, cobraSeed: int, feval=300, conTol: float|None=0):
        """
            Perform SACOBRA-runs with method ``meth`` on multiple G-problems and multiple runs:

            - ``meth = 'one_s'``: One SACOBRA configuration for all G-problems,
            - ``meth = 'solve'``: G-problem-specific SACOBRA configuration (see ex_COP.py)

        :param gnames:  list of G-problem names
        :param dims:    list of corresponding dimensions (only relevant for G02 and G03)
        :param runs:    how many runs
        :param cobraSeed: run ``r in range(runs)`` gets seed ``cobraSeed + r``
        :param feval:   budget of real function evaluations
        :param conTol:  common constraint tolerance for all runs (if None, use the defaults of each (gname,meth)-combi)
        :return:        a data frame with one row for each run and columns 'time' (computation time in ms), 'err' (final
                        error after feval iterations) and others
        """
        cop = ExamCOP()        # is used indirectly below in eval(...)
        df2 = pd.DataFrame()
        for i, gname in enumerate(gnames):
            dim = dims[i]
            for meth in ['one_s',]:   #  'solve',
                for run in range(runs):
                    start = time.perf_counter()
                    if conTol is None:          # use the default conTol of each method
                        conTolStr = ""
                    else:
                        conTolStr = f", conTol={conTol}"

                    #     if meth == 'one_s':
                    #         c2 = self.one_s(gname, dim, cobraSeed + run, feval)
                    #     else:   # i.e. if meth=='solve'
                    #         if gname in {"G02", "G03"}:
                    #             c2 = eval(f"cop.solve_{gname}(cobraSeed + run, {dim}, feval={feval}, verbIter=100)")
                    #         else:
                    #             c2 = eval(f"cop.solve_{gname}(cobraSeed + run, feval={feval}, verbIter=100)")
                    # else:
                    if meth == 'one_s':
                        c2 = eval(f"self.one_s(gname, dim, cobraSeed + run, feval {conTolStr})")
                    else:   # i.e. if meth=='solve'
                        if gname in {"G02", "G03"}:
                            c2 = eval(f"cop.solve_{gname}(cobraSeed + run, {dim}, feval={feval}, verbIter=100 {conTolStr})")
                        else:
                            c2 = eval(f"cop.solve_{gname}(cobraSeed + run, feval={feval}, verbIter=100 {conTolStr})")
                    time_ms = (time.perf_counter() - start) / runs * 1000
                    fin_err = c2.p2.fin_err
                    new_row_df2 = pd.DataFrame(
                        {
                            'gname': gname,
                            'd': c2.p2.dim,
                            'meth': meth,
                            'seed': cobraSeed + run,
                            'time': time_ms,
                            'err': fin_err,
                            'feval': feval,
                            'conTol': c2.p2.conTol}, index=[0])
                    df2 = pd.concat([df2, new_row_df2], axis=0)
        print(df2)
        df2.to_feather("feather/df2.feather")
        print(f"df2 saved to {os.getcwd()}/feather/df2.feather")
        print(f"\n Number of runs in df2: {df2.seed.unique().size} for each (gname,meth)-combi")
        print("\n --- Median for each problem --- ")
        print(df2.groupby(['gname','meth']).median())
        print("\n ---  Std for each problem --- ")
        print(df2.groupby(['gname','meth']).std())
        return df2

    def multi_init(self, gnames: list, cobraSeed: int, feval=0):
        """
            Perform SACOBRA-inits on multiple G-problems.

            Within the 13 problems G01, ..., G13, only G05 and G10 activate constraint normalization
        """
        init_df = pd.DataFrame()
        for gname in gnames:
            start = time.perf_counter()
            c2 = self.one_s(gname, cobraSeed, feval)
            time_ms = (time.perf_counter() - start) * 1000
            fin_err = c2.p2.fin_err
            new_row_df = pd.DataFrame(
                {
                    'gname': gname,
                    'd': c2.p2.dim,
                    'GR': c2.cobra.sac_res['GR'],
                    'TGR': c2.cobra.sac_opts.ISA.TGR,
                    'seed': cobraSeed,
                    'time': time_ms,
                    'err': fin_err}, index=[0])
            init_df = pd.concat([init_df, new_row_df], axis=0)
        print(init_df)
        init_df.to_feather("feather/init_df.feather")
        print(f"init_df saved to {os.getcwd()}/feather/init_df.feather")
        return init_df

    def df_analyze(self, fname1, fname2=None):
        df1 = pd.read_feather("feather/"+fname1)
        nrun = df1.seed.unique().size
        nmeth = df1.meth.unique().size
        ngname = df1.gname.unique().size
        print(f"\n Number of runs in df1: {nrun} for each (gname,meth)-combi")
        # assert nrun*nmeth*ngname == df1.shape[0]
        if fname2 is not None:
            # this is just to compare 'time' and 'err' from df1 (e.g. a run with conTol=0.0) with
            # 'time2' and 'err2' from df2 (e.g. a run with conTol=1e-7) in a row-by-row fashion.
            #
            # It turns out that results for conTol = 0.0 | 1e-7 are very similar (at least for G01, ..., G13).
            #
            df2 = pd.read_feather("feather/"+fname2)
            assert np.all(df1['gname'] == df2['gname'] )
            assert np.all(df1['meth'] == df2['meth'])
            df1['time2'] = df2['time']
            df1['err2'] = df2['err']
            df1 = df1.drop(["conTol","seed"],axis=1)    # drop some columns so that all other columns get printed
        print("\n --- Median for each (problem, meth) --- ")
        print(df1.groupby(['gname', 'meth', 'd']).median())
        print("\n ---  Std for each (problem, meth) --- ")
        print(df1.groupby(['gname', 'meth', 'd']).std())
        # print("\n --- Mean for each problem --- ")
        # del df1['meth']
        # print(df1.groupby(['gname']).mean())

        print(f"\nThe (G02, d=2)-errors for {fname1}:")
        G02_2_errs = np.array(df1[(df1["gname"]=="G02") &
                                  (df1["d"]==2) &
                                  (df1["meth"]=="one_s")]["err"])
        print(np.sort(G02_2_errs))
        print(f"median = {np.median(G02_2_errs)}")

        print(f"\nThe (G03, d=10)-errors for {fname1}:")
        G03_10_errs = np.array(df1[(df1["gname"]=="G03") & (df1["d"]==10)]["err"])
        print(np.sort(G03_10_errs))
        print(f"median = {np.median(G03_10_errs)}")


if __name__ == '__main__':
    one = OneS()
    gnames = ["G01", "G02", "G02", "G03", "G03", "G04", "G05", "G06", "G07", "G08", "G09", "G10", "G11", "G12", "G13"]
    dims   = [   -1,     2,     5,     7,    10,   -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1]
    gnames = ["G03", "G09",]  #  "G08", "G09", "G10",
    dims   = [   10,   -1]
    # gnames = ["G13"]  # "G10", "G11", "G12",
    # dims   = [ -1]  #   -1,    -1,    -1,
    df2 = one.one_s_multi_g_r(gnames, dims,10, 54, feval=500, conTol=0)       # conTol=0 | 1e-7
    # init_df = one.multi_init(gnames, 54, feval=120)
    # one.df_analyze("df2_conTol0.0-fe500-G01-G13.feather", "df2_conTol1e-7-fe500-G01-G13.feather")
    # one.df_analyze("df2_conTol0.0-fe500-G02-d02.feather")
    # one.df_analyze("df2_conTol0.0-MIDPTS-gauss-fe500-G01-G13.feather")   # NONE | XNEW | MIDPTS




