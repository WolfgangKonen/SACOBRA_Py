import time
import numpy as np
import pandas as pd

from cobraInit import CobraInitializer
from ex_COP import analyze_solution
from gCOP import GCOP, show_error_plot
from cobraPhaseII import CobraPhaseII
from opt.equOptions import EQUoptions
from opt.isaOptions import ISAoptions, O_LOGIC
from opt.sacOptions import SACoptions
from opt.idOptions import IDoptions
from opt.rbfOptions import RBFoptions
from opt.seqOptions import SEQoptions

verb=1

def solve_G17(cobraSeed, feval=500, kernel="cubic", muFinal=1e-4, conTol=0.0):  # conTol=1e-7
    """ Test whether COP G17 has statistical equivalent results to the R side with squares=T, if we set on the
        Python side RBF.degree=2 (which is similar, but not the same).

        We test that the median of 15 final errors is < 1e-13, which is statistically equivalent to the R side
        (see ex_COP.R, function solve_G17, multi_gfnc)
    """
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
    return c2

if __name__ == '__main__':
    c2 = solve_G17(cobraSeed=61, feval=500, kernel="cubic",    muFinal=1e-4, conTol=0)
    analyze_solution(c2, c2.cobra)
    c2 = solve_G17(cobraSeed=61, feval=500, kernel="gaussian", muFinal=1e-4, conTol=0)
    analyze_solution(c2, c2.cobra)
    c2 = solve_G17(cobraSeed=61, feval=500, kernel="cubic",    muFinal=1e-4, conTol=1e-7)
    analyze_solution(c2, c2.cobra)
    c2 = solve_G17(cobraSeed=61, feval=500, kernel="gaussian", muFinal=1e-4, conTol=1e-7)
    analyze_solution(c2, c2.cobra)
