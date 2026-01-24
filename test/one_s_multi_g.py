import os
import pickle
import time
import numpy as np
import pandas as pd
from datetime import datetime

from cobraInit import CobraInitializer
from ex_COP import ExamCOP
from gCOP import GCOP, png_error_plot, show_error_plot
from cobraPhaseII import CobraPhaseII
from opt.equOptions import EQUoptions
from opt.isaOptions import ISAoptions, O_LOGIC
from opt.riOptions import RIoptions
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
        :param dim:         dimension for G-problems with variable dimension
        :param cobraSeed:   seed
        :param feval:       real function evaluations
        :param conTol:      constraint tolerance, common values are 0 or 1e-7
        :return:    ``c2``, the resulting object after running ``CobraPhaseII.start()``
        """
        print(f"Starting one_s({gname}, dim={dim}, {cobraSeed}) ...")
        muFinal = 1e-4   # 1e-4 | 1e-7
        if gname in {"G02", "G03"}:
            gcop = GCOP(gname, dimension=dim, mu=muFinal)
        else:
            gcop = GCOP(gname, mu=muFinal)

        dim = gcop.dimension
        idp = (dim + 1) * (dim + 2) // 2
        if feval == 0: feval = idp+2

        equ = EQUoptions(muGrow=100, muDec=1.6, muFinal=muFinal,
                         refinePrint=False, refineAlgo="L-BFGS-B")  # "L-BFGS-B COBYLA"
        cobra = CobraInitializer(gcop.x0, gcop.fn, gcop.name, gcop.lower, gcop.upper, gcop.is_equ,
                                 solu=gcop.solu,
                                 s_opts=SACoptions(verbose=verb, verboseIter=100, feval=feval, cobraSeed=cobraSeed,
                                                   ID=IDoptions(initDesign="LHS", initDesPoints=idp),
                                                   RBF=RBFoptions(degree=2),   # for default interpolator="scipy" + "cubic"
                                                   # RBF=RBFoptions(degree=1.5, interpolator="sacobra"),  # test only, "cubic"
                                                   # RBF=RBFoptions(kernel="gaussian", degree=2),   # alternative "gaussian"
                                                   # ISA=ISAoptions(onlinePLOG=O_LOGIC.NONE),   # the default (before 2025/08/01)
                                                   ISA=ISAoptions(onlinePLOG=O_LOGIC.MIDPTS), # run 2025/08/12   # , TGR=np.inf
                                                   # ISA=ISAoptions(onlinePLOG=O_LOGIC.XNEW),     # run 2025/08/13
                                                   EQU=equ,
                                                   RI=RIoptions(repairInfeas=True, eps2=0, q=3, repairMargin=np.inf, checkIt=False), # new 2025/10/15
                                                   SEQ=SEQoptions(finalEpsXiZero=True,  # epsilonMax=0.0,
                                                                  conTol=conTol,)))  #,  trueFuncForSurrogates=True
        # --- ncall-debug only: ---
        # print(f"after cobraInit: gcop.ncall = {gcop.ncall}")
        # cobra.sac_res['ncall'][18] = idp                # initial design points
        # cobra.sac_res['ncall'][19] = gcop.ncall - idp   # remaining calls during cobraInit

        if feval > idp:
            c2 = CobraPhaseII(cobra).start(gcop)
            # will also set various variables in c2.p2 via p2.fill()

        if c2.p2.fin_err < 1e-7:
            dummy = 0
        print(f"final err: {c2.p2.fin_err}")
        # show_error_plot(cobra, gcop, c2.get_muVec())  #  ylim=[1e-4,1e0]
        # print(gcop.fn(gcop.solu))
        print(gcop.fbest)
        print(c2.cobra.get_fbest())
        #
        # --- ncall-debug only: ---
        # print(f"after phase II: gcop.ncall = {gcop.ncall}")
        # ind_ncall = np.flatnonzero(cobra.sac_res['ncall']>0)
        # print(f"sum(sac_res['ncall'] = {sum(cobra.sac_res['ncall'])} from inidices "
        #       f"{ind_ncall} with values {cobra.sac_res['ncall'][ind_ncall]}" )
        # print(f"missing calls: {gcop.ncall - sum(cobra.sac_res['ncall']) }")

        return c2

    def one_s_multi_g_r(self, gnames: list, dims: list, runs: int, cobraSeed: int, feval=300, conTol: float|None=0):
        """
            Perform multiple SACOBRA-runs on multiple G-problems with method ``meth`` (def'd in source code below):

            - ``meth='one_s'``: One SACOBRA configuration for all G-problems,
            - ``meth='solve'``: G-problem-specific SACOBRA configuration (see ExamCop)

        :param gnames:  list of G-problem names
        :param dims:    list of corresponding dimensions (only relevant for G02 and G03)
        :param runs:    how many runs
        :param cobraSeed: run ``r in range(runs)`` gets seed ``cobraSeed + r``
        :param feval:   budget of real function evaluations
        :param conTol:  common constraint tolerance for all runs (if None, use the defaults of each (gname,meth)-combi)
        :return:        a data frame, with one row for each run and columns 'time' (computation time in ms), 'err' (final
                        error after feval iterations) and others, which is also saved to
                        ``"feather/df2.feather"``.
        """
        current_datetime = datetime.now()
        dir_run = current_datetime.strftime("feather/run%Y-%m-%d_%Hh%Mm%S")
        if not os.path.exists(dir_run):
            os.mkdir(dir_run)
        cop = ExamCOP()        # is used indirectly below in eval(...)
        dfsum = pd.DataFrame()
        for i, gname in enumerate(gnames):
            dim = dims[i]
            for meth in ['solve',]:   #  'solve','one_s'
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
                        # why 'eval(...)'? - to be able to call cop.solve_{*} and to add conTolStr in a flexible way
                    else:   # i.e. if meth=='solve'
                        if gname in {"G02", "G03"}:
                            c2 = eval(f"cop.solve_{gname}(cobraSeed + run, {dim}, feval={feval}, verbIter=100 {conTolStr})")
                        else:
                            c2 = eval(f"cop.solve_{gname}(cobraSeed + run, feval={feval}, verbIter=100 {conTolStr})")
                    time_ms = (time.perf_counter() - start) / runs * 1000
                    new_row_dfs = pd.DataFrame(
                        {
                            'gname': gname,
                            'd': c2.p2.dim,
                            'meth': meth,
                            'seed': cobraSeed + run,
                            'time': time_ms,
                            'err': c2.p2.fin_err,
                            'feval': feval,
                            'conTol': c2.p2.conTol,
                            'maxViol': c2.p2.maxViol,
                            'isFeas': 0 if c2.p2.maxViol > 0 else 1,
                            'maxConstr': max(c2.p2.constr),     # c2.p2.constr: see end of CobraPhaseII::start
                            'ncall': c2.p2.ncall,
                            'n_repair': c2.p2.ri2.n_repair,
                            'n_rep_suc': c2.p2.ri2.n_rep_suc,
                        }, index=[0])
                    dfsum = pd.concat([dfsum, new_row_dfs], axis=0)
                    f_prefix = f"{dir_run}/{gname}_{c2.p2.dim:02d}_{run:02d}"
                    c2_df1 = c2.cobra.df.drop(["optimizer", "optimConv"], axis=1)
                    c2_df2 = c2.cobra.df2.drop(["predSoluPenal", "sigmaD", "penaF", "err1", "err2",
                                               "nv_cB", "nv_cA", "nv_tB", "nv_tA"], axis=1)
                    c2_df1.to_feather(f"{f_prefix}_df1.feather")
                    print(f"c2.cobra.df  saved to {os.getcwd()}/{f_prefix}_df1.feather")
                    c2_df2.to_feather(f"{f_prefix}_df2.feather")
                    print(f"c2.cobra.df2 saved to {os.getcwd()}/{f_prefix}_df2.feather")
                    gtitle = f"{gname}, d={c2.p2.dim:02d}"
                    png_error_plot(c2.cobra.df, c2.get_muVec(), c2.p2.f_solu, gtitle, png_file=f"{f_prefix}.png")
                    dummy = 0
        print(dfsum)
        dfsum.to_feather(f"{dir_run}/dfsum.feather")
        print(f"dfsum saved to {os.getcwd()}/{dir_run}/dfsum.feather")
        # --- Read it back with: ---
        # dfsum = pd.read_feather(f"{dir_run}/dfsum.feather")
        dfsum.to_csv(f"{dir_run}/dfsum.csv", sep=";", index = False, float_format =" %.8e")
        print(f"\n Number of runs in dfsum: {dfsum.seed.unique().size} for each (gname,meth)-combi")
        print("\n --- Median for each problem --- ")
        # if there are NaNs in column dfsum['err'] (run with no feasible solu found), then groupby will automatically
        # drop all NaN-rows prior to median calculation. The number of feasible runs is found by summing column
        # dfsum['isFeas'] in s_df and replacing this column in m_df by the s_df-column
        m_df = dfsum.groupby(['gname','meth','d']).median()
        s_df = dfsum.groupby(['gname','meth','d']).sum()
        m_df['isFeas'] = s_df['isFeas']
        m_df = m_df.drop(["seed", "conTol", "maxViol", "n_repair", "n_rep_suc"], axis=1)
        print(m_df)
        print("\n ---  Std for each problem --- ")
        s_df = dfsum.groupby(['gname','meth','d']).std()
        s_df = s_df.drop(["seed", "conTol", "maxViol", "n_repair", "n_rep_suc"], axis=1)
        print(s_df)
        m_df['std_time'] = s_df['time']
        m_df['std_err'] = s_df['err']
        m_df.to_csv(f"{dir_run}/med_std_grp.csv", sep=";", index=True, float_format=" %.8e")
        with open(f"{dir_run}/s_opts.pickle", 'wb') as f:
            pickle.dump(c2.cobra.sac_opts, f, pickle.HIGHEST_PROTOCOL)
        print(f"sac_opts saved to {os.getcwd()}/{dir_run}/s_opts.pickle")
        # --- Read it back with: ---
        # with open(f"{dir_run}/s_opts.pickle", 'rb') as f:
        #     s_opts = pickle.load(f)

        return dfsum

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
            new_row_df = pd.DataFrame(
                {
                    'gname': gname,
                    'd': c2.p2.dim,
                    'GR': c2.cobra.sac_res['GR'],
                    'TGR': c2.cobra.sac_opts.ISA.TGR,
                    'seed': cobraSeed,
                    'time': time_ms,
                    'err': c2.p2.fin_err}, index=[0])
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
        x = df1.groupby(['gname', 'meth', 'd']).median()   # median() will automatically drop NaNs (!)
        print(x.loc[:, ['time', 'err', 'n_repair', 'n_rep_suc']])
        print("\n ---  Std for each (problem, meth) --- ")
        x = df1.groupby(['gname', 'meth', 'd']).std()
        if 'maxViol' in x.columns:
            print(x.loc[:, ['err','maxViol','n_repair','n_rep_suc']])                    # to get it printed if df1 has too many columns
        else:
            print(x.loc[:, ['err','n_repair','n_rep_suc']])
        # print("\n --- Mean for each problem --- ")
        # del df1['meth']
        # print(df1.groupby(['gname']).mean())
        y = df1.groupby(['gname', 'meth', 'd']).count()       # count() will automatically drop NaNs
        print("\n ---  Runs that found no feasible solution --- ")        # --> err may have less counts due to NaNs for
        print(y['feval'] - y['err'])                          # 'infeasible runs'

        if any(df1['gname'] == "G02"):
            print(f"\nThe (G02, d=2)-errors for {fname1}:")
            G02_2_errs = np.array(df1[(df1["gname"]=="G02") &
                                      (df1["d"]==2) &
                                      (df1["meth"]=="one_s")]["err"])
            print(np.sort(G02_2_errs))
            print(f"median = {np.median(G02_2_errs)}")

        if any(df1['gname'] == "G03"):
            print(f"\nThe (G03, d=10)-errors for {fname1}:")
            G03_10_errs = np.array(df1[(df1["gname"]=="G03") & (df1["d"]==10)]["err"])
            print(np.sort(G03_10_errs))
            print(f"median = {np.median(G03_10_errs)}")


if __name__ == '__main__':
    one = OneS()
    gnames = ["G03", "G09",]  #  "G08", "G09", "G10",
    dims   = [   10,   -1]
    gnames = ["G22"]  # , "G21", "G22", "G21", "G22", "G24" "G10", "G11", "G12",
    dims   = [   -1]  # ,    -1,    -1,    -1,    -1,    -1,
    gnames = ["G05", "G06"]     # "G13",
    dims   = [   -1,    -1]  # ,    -1,    -1,    -1,    -1,
    gnames = ["G05", "G06"]     # "G13",
    dims   = [   -1,    -1]  # ,    -1,    -1,    -1,    -1,
    gnames = ["G02"]  # , "G21", "G22", "G21", "G22", "G24" "G10", "G11", "G12",
    dims   = [   2]  # ,    -1,    -1,    -1,    -1,    -1,
    gnames = ["G01", "G02", "G02", "G03", "G03", "G04", "G05", "G06", "G07", "G08", "G09", "G10", "G11", "G12", "G13"] #
    dims   = [  -1,     2,     5,     7,    10,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1 ] #
    gnames = ["G05", "G06"]  # , "G13",
    dims   = [   -1,    -1]  # ,    -1,    -1,    -1,    -1,
    gnames = ["G14", "G15", "G16", "G17", "G18", "G19", "G21", "G22", "G23", "G24"]
    dims   = [  -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1]
    gnames = ["G17"]  # , "G21", "G22", "G21", "G22", "G24" "G10", "G11", "G12",
    dims   = [   -1]  # ,    -1,    -1,    -1,    -1,    -1,
    df2 = one.one_s_multi_g_r(gnames, dims,3, 65, feval=500, conTol=0.0)     #   # conTol=1e-4 | 1e-7
    # init_df = one.multi_init(gnames, 54, feval=120)
    # one.df_analyze("df2_conTol0.0-fe500-G01-G13.feather", "df2_conTol1e-7-fe500-G01-G13.feather")
    # one.df_analyze("df2_conTol0.0-MIDPTS-fe500-G14-G24.feather")   # NONE | XNEW | MIDPTS
    # one.df_analyze("df2_conTol0.0-MIDPTS-fe500-G14-G24-EPS0.feather")
    # one.df_analyze("df2.feather")
    # one.df_analyze("df2-G22-T5.feather")
    # one.df_analyze("df2_repair-conTol0.0-MIDPTS-cubic-fe500-G14-24.feather")




