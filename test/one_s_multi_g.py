import os
import time
import numpy as np
import pandas as pd

from cobraInit import CobraInitializer
from ex_COP import ExamCOP
from gCOP import GCOP, show_error_plot
from cobraPhaseII import CobraPhaseII
from opt.equOptions import EQUoptions
from opt.isaOptions import ISAoptions
from opt.sacOptions import SACoptions
from opt.idOptions import IDoptions
from opt.rbfOptions import RBFoptions
from opt.seqOptions import SEQoptions

verb = 1
class OneS:
    def one_s(self, gname: str, cobraSeed: int, feval=300):
        """ Test whether COP G21 has statistical equivalent results to the R side with squares=T, if we set on the
            Python side RBF.degree=2 (which is similar, but not the same).

            We test that the median of 15 final errors is < 1e-13, which is statistically equivalent to the R side
            (see ex_COP.R, function solve_G17, multi_gfnc)
        """
        print(f"Starting one_s({gname}, {cobraSeed}) ...")
        if gname == "G02":
            gcop = GCOP(gname, dimension=2)
        elif gname == "G03":
            gcop = GCOP(gname, dimension=7)
        else:
            gcop = GCOP(gname)

        dim = gcop.dimension
        idp = (dim + 1) * (dim + 2) // 2
        if feval==0: feval=idp+2

        equ = EQUoptions(muGrow=100, muDec=1.6, muFinal=1e-7,
                         refinePrint=False, refineAlgo="L-BFGS-B")  # "L-BFGS-B COBYLA"
        cobra = CobraInitializer(gcop.x0, gcop.fn, gcop.name, gcop.lower, gcop.upper, gcop.is_equ,
                                 solu=gcop.solu,
                                 s_opts=SACoptions(verbose=verb, verboseIter=100, feval=feval, cobraSeed=cobraSeed,
                                                   ID=IDoptions(initDesign="LHS", initDesPoints=idp),
                                                   RBF=RBFoptions(degree=2),
                                                   EQU=equ,
                                                   SEQ=SEQoptions(finalEpsXiZero=True, conTol=0)))   # conTol= 0 | 1e-7
        if feval > idp: c2 = CobraPhaseII(cobra).start()

        fin_err = np.array(cobra.get_fbest() - gcop.fbest)
        c2.p2.fin_err = fin_err
        print(f"final err: {fin_err}")
        c2.p2.fe_thresh = 1e-1
        c2.p2.dim = gcop.dimension
        # show_error_plot(cobra, gcop,)  #  ylim=[1e-4,1e0]
        # print(gcop.fn(gcop.solu))
        print(gcop.fbest)
        print(c2.cobra.get_fbest())
        return c2


    def one_s_multi_g_r(self, gnames: list, runs: int, cobraSeed: int, feval=300):
        """
            Perform SACOBRA-runs with method ``meth`` on multiple G-problems and multiple runs:

            - ``meth = 'one_s'``: One SACOBRA configuration for all G-problems,
            - ``meth = 'solve'``: G-problem-specific SACOBRA configuration (see ex_COP.py)
        """
        cop = ExamCOP();        # is used indirectly below in eval(...)
        df2 = pd.DataFrame()
        for gname in gnames:
            for meth in [ 'one_s', 'solve',]:   #
                for run in range(runs):
                    start = time.perf_counter()
                    if meth == 'one_s':
                        c2 = self.one_s(gname, cobraSeed + run, feval)
                    else:   # i.e. if meth=='solve'
                        c2 = eval(f"cop.solve_{gname}(cobraSeed + run, feval={feval}, verbIter=100)")
                    time_ms = (time.perf_counter() - start) / runs * 1000
                    fin_err = c2.p2.fin_err
                    new_row_df2 = pd.DataFrame(
                        {
                            'gname': gname,
                            'd': c2.p2.dim,
                            'meth': meth,
                            'seed': cobraSeed + run,
                            'time': time_ms,
                            'err': fin_err}, index=[0])
                    df2 = pd.concat([df2, new_row_df2], axis=0)
        print(df2)
        df2.to_feather("df2.feather")
        print(f"df2 saved to {os.getcwd()}/df2.feather")
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
        init_df.to_feather("init_df.feather")
        print(f"init_df saved to {os.getcwd()}/init_df.feather")
        return init_df

    def df_analyze(self, fname1, fname2=None):
        df1 = pd.read_feather(fname1)
        if fname2 is not None:
            df2 = pd.read_feather(fname2)
            assert np.all(df1['gname'] == df2['gname'] )
            assert np.all(df1['meth'] == df2['meth'])
            df1['time2'] = df2['time']
            df1['err2'] = df2['err']
        print("\n --- Median for each problem --- ")
        print(df1.groupby(['gname', 'meth']).median())
        print("\n ---  Std for each problem --- ")
        print(df1.groupby(['gname', 'meth']).std())
        # print("\n --- Mean for each problem --- ")
        # del df1['meth']
        # print(df1.groupby(['gname']).mean())

if __name__ == '__main__':
    one = OneS()
    gnames = ["G01", "G02", "G03", "G04", "G05", "G06", "G07", "G08", "G09", "G10", "G11", "G12", "G13"]
    # gnames = ["G07", "G08", "G09", "G10", ]  #
    # df2 = one.one_s_multi_g_r(gnames, 5, 54, feval=500)
    init_df = one.multi_init(gnames, 54, feval=120)
    # one.df_analyze("df2_conTol0.0-fe500-G01-G13.feather", "df2_conTol1e-7-fe500-G01-G13.feather")



