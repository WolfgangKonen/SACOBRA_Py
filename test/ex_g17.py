import os
import pickle
import time
import numpy as np
import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame
from datetime import datetime

from cobraInit import CobraInitializer
from ex_COP import analyze_solution, plot_func_surr
from gCOP import GCOP, show_error_plot
from cobraPhaseII import CobraPhaseII
from opt.equOptions import EQUoptions
from opt.isaOptions import ISAoptions, O_LOGIC
from opt.sacOptions import SACoptions
from opt.idOptions import IDoptions
from opt.rbfOptions import RBFoptions
from opt.seqOptions import SEQoptions

verb=1


def append_df3_new_row(df3: DataFrame, c2: CobraPhaseII, cobra: CobraInitializer, kernel, muFinal, conTol):
    """
    Append a new row to data frame ``df3``
    :return: ``df3``
    """
    fn_xbest, fn_solu, fsurr_xbest, fsurr_solu, csurr_xbest, csurr_solu, max_d_solu \
        = analyze_solution(c2, cobra)
    # csurr_xbest|solu are the constraint surrogate values at input point xbest or (true) solu, resp.
    # For equality constraints, take abs() and subtract muFinal. Then, a value > conTol indicates constraint violation
    # c_xbest|solu is the same for the true constraint values.
    equ_ind = np.flatnonzero(cobra.sac_res['is_equ'])
    csurr_xbest[:, equ_ind] = abs(csurr_xbest[:, equ_ind]) - muFinal
    csurr_solu[:, equ_ind] = abs(csurr_solu[:, equ_ind]) - muFinal
    c_xbest = fn_xbest[1:].copy()
    c_solu = fn_solu[1:].copy()
    c_xbest[equ_ind] = abs(c_xbest[equ_ind]) - muFinal
    c_solu[equ_ind] = abs(c_solu[equ_ind]) - muFinal
    new_row_df3 = pd.DataFrame(
        {
            'kernel': kernel,
            'muFinal': muFinal,
            'conTol': conTol,
            'max_d_solu': max_d_solu,       # max. deviation in diff vector xbest - solu
            'err': c2.p2.fin_err,           # final error in objective
            'f_xbest': fn_xbest[0],         # objective value at xbest
            'fsurr_xbest': fsurr_xbest,     # objective surrogate value at xbest
            'f_solu': fn_solu[0],           # objective value at (true) solution
            'fsurr_solu': fsurr_solu,       # objective surrogate value at (true) solution
            'max_c_xbest': np.max(c_xbest), # maximum (true) constraint value at xbest
            'max_c_solu': np.max(c_solu),   # maximum (true) constraint value at (true) solution
            'csurr_xbest': np.max(csurr_xbest), # maximum constraint surrogate value at xbest
            'csurr_solu': np.max(csurr_solu),   # maximum constraint surrogate value at (true) solution
            'is_feas': 0 if c2.p2.maxViol > 0 else 1,   # 1 if xbest is feasible, 0 if not
        }, index=[0])
    df3 = pd.concat([df3, new_row_df3], axis=0)
    return df3


def solve_G05(df3: DataFrame, dir_run, cobraSeed, feval=170, kernel="cubic", muFinal=1e-4, conTol=0):  # conTol=0 | 1e-7
    print(f"Starting solve_G05({cobraSeed}) ...")
    gcop = GCOP("G05")
    idp = 15  # =(d+1)(d+2)/2, the minimum for RBF.kernel="cubic", RBF.degree=2 and d=4

    cobra = CobraInitializer(gcop.x0, gcop.fn, gcop.name, gcop.lower, gcop.upper, gcop.is_equ,
                             solu=gcop.solu,
                             s_opts=SACoptions(verbose=verb, verboseIter=100, feval=feval, cobraSeed=cobraSeed,
                                               ID=IDoptions(initDesign="LHS", initDesPoints=idp),
                                               RBF=RBFoptions(degree=2, kernel=kernel),
                                               EQU=EQUoptions(muDec=1.6, muFinal=muFinal, refinePrint=False,
                                                              refineAlgo="COBYLA"),  # "L-BFGS-B COBYLA"
                                               SEQ=SEQoptions(finalEpsXiZero=True, conTol=conTol)))
    c2 = CobraPhaseII(cobra).start(gcop=gcop)
    f_prefix = f"{dir_run}/{gcop.name}_{c2.p2.dim:02d}_k{kernel}_mu1m{int(-np.log10(muFinal)):02d}"

    print(f"final err: {c2.p2.fin_err}")
    print(c2.p2.f_solu)
    print(c2.cobra.get_fbest())
    c2.p2.fe_thresh = 5e-6

    for dim in range(gcop.dimension):
        for icomp in range(cobra.sac_res['nConstraints']+1):
            png_file = f"{f_prefix}_i{icomp}_d{dim}.png"
            plot_func_surr(c2, cobra, icomp, dim, 30, png_file)

    df3 = append_df3_new_row(df3, c2, cobra, kernel, muFinal, conTol)

    return c2, df3


def solve_G17(df3: DataFrame, dir_run, cobraSeed, feval=500, kernel="cubic", muFinal=1e-4, conTol=0.0):  # conTol=1e-7
    print(f"\n--- kernel={kernel}, muFinal={muFinal}, conTol={conTol} ---")
    print(f"Starting solve_G17({cobraSeed}) ...")
    G17 = GCOP("G17", mu=muFinal)
    dim = G17.dimension
    idp = (dim + 1) * (dim + 2) // 2

    equ = EQUoptions(muGrow=100, muDec=1.6, muFinal=muFinal,
                     refinePrint=False, refineAlgo="COBYLA")  # "L-BFGS-B COBYLA"
    cobra = CobraInitializer(G17.x0, G17.fn, G17.name, G17.lower, G17.upper, G17.is_equ,
                             solu=G17.solu,
                             s_opts=SACoptions(verbose=verb, verboseIter=100, feval=feval, cobraSeed=cobraSeed,
                                               ID=IDoptions(initDesign="LHS", initDesPoints=idp),
                                               RBF=RBFoptions(degree=2, kernel=kernel),  #
                                               EQU=equ,
                                               ISA=ISAoptions(onlinePLOG=O_LOGIC.MIDPTS),
                                               SEQ=SEQoptions(finalEpsXiZero=True, conTol=conTol)))

    c2 = CobraPhaseII(cobra).start(gcop=G17)

    print(f"final err: {c2.p2.fin_err}")
    print(c2.p2.f_solu)
    print(c2.cobra.get_fbest())
    c2.p2.fe_thresh = 1e-13

    # c2_save = c2.copy()
    # c2_save.cobra.sac_res['fn'] = None
    # c2_save.cobra.sac_res['originalfn'] = None
    # c2_save.cobra.rw = None
    # c2_save.cobra.ri2 = None
    # with open(f"feather/current_c2.pickle", 'wb') as f:
    #     pickle.dump(c2_save, f, pickle.HIGHEST_PROTOCOL)

    plot_func_surr(c2, cobra, 0, 1, 20)

    df3 = append_df3_new_row(df3, c2, cobra, kernel, muFinal, conTol)

    return c2, df3

def analyze_current_c2():
    # --- not yet working ---
    with open(f"feather/current_c2.pickle", 'rb') as input:
        c2 = pickle.load(input)
    cobra = c2.cobra
    fn_xbest, fn_solu, fsurr_xbest, fsurr_solu, csurr_xbest, csurr_solu, max_d_solu \
        = analyze_solution(c2, cobra)
    plot_func_surr(c2, cobra, 0, 1, 20)
    dummy = 0


if __name__ == '__main__':
    # analyze_current_c2()
    df3 = pd.DataFrame()
    current_datetime = datetime.now()
    dir_run = current_datetime.strftime("feather/run%Y-%m-%d_%Hh%Mm%S")
    if not os.path.exists(dir_run):
        os.mkdir(dir_run)

    c2, df3 = solve_G05(df3, dir_run, cobraSeed=61, feval=100, kernel="gaussian",    muFinal=1e-4, conTol=0)
    c2, df3 = solve_G17(df3, dir_run, cobraSeed=61, feval=500, kernel="cubic",    muFinal=1e-4, conTol=0)
    # c2, df3 = solve_G17(df3, dir_run, cobraSeed=61, feval=500, kernel="cubic",    muFinal=1e-4, conTol=1e-7)
    c2, df3 = solve_G17(df3, dir_run, cobraSeed=61, feval=500, kernel="gaussian", muFinal=1e-4, conTol=0)
    # c2, df3 = solve_G17(df3, dir_run, cobraSeed=61, feval=500, kernel="gaussian", muFinal=1e-4, conTol=1e-7)
    c2, df3 = solve_G17(df3, dir_run, cobraSeed=61, feval=500, kernel="gaussian", muFinal=1e-7, conTol=0)
    df3.to_csv(f"feather/g17_df3.csv", sep=";", index=True, float_format=" %.8e")
    dummy = 0

