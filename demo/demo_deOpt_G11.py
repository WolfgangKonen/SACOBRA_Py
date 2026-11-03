# Demonstrate that scipy's differential evolution can solve COP G11.
# Compare demo solver run_demo_G11() with SACOBRA-like solver run_seq_opt_G11()
#
# Results (printed below): not exactly the same (time, feMax, ncall), but exactly same solution, objective and err
#
import time
import numpy as np
from scipy.optimize import differential_evolution, Bounds, LinearConstraint, NonlinearConstraint

from cobraInit import CobraInitializer
from cobraPhaseII import CobraPhaseII
from gCOP import GCOP
from opt.equOptions import EQUoptions
from opt.idOptions import IDoptions
from opt.rbfOptions import RBFoptions
from opt.sacOptions import SACoptions
from opt.seqOptions import SEQoptions
from seqFuncFactory import check_if_cobra_optimizable
from seqOptimizer import SeqOptimizer
from surrogator import Surrogator

conTol = 0.0  # 1e-5  #
muFinal = 1e-4
cobraSeed = 42
ncall, ncall_h = 0, 0


def run_demo_G11():
    """
    Demo solver for COP G11, self-contained
    """
    def g11_fn(x):
        return np.array([x[0] ** 2 + (x[1] - 1) ** 2,
                         x[1] - x[0] ** 2])

    # Define the objective function
    def objective_function(x):
        global ncall
        ncall += 1
        return x[0] ** 2 + (x[1] - 1) ** 2

    # Define variable bounds
    G21_lower = [-1, -1]
    G21_upper = [1, 1]
    G21_solu = np.array([-np.sqrt(0.5-muFinal), 0.5])        # obj:  0.75 - muFinal, maxViol: muFinal
    lower = G21_lower
    upper = G21_upper
    bounds = Bounds(lower, upper)

    # xStart = np.array([0.20134874, 0.29918661])
    xStart = np.array([-0.26015247,  0.84666318])     # copied from xStart in run_seq_opt_G11()

    # # Define a nonlinear inequality constraint: x[0]**2 + x[1] <= 5
    # def g1_func(x):
    #     return -x[0] + 35 * (x[1] ** (0.6)) + 35 * (x[2] ** 0.6)
    # nl_g1 = NonlinearConstraint(g1_func, -np.inf, conTol)

    # Define nonlinear equality constraints: -muFinal <= cond(x) <= muFinal
    def h1_func(x):
        global ncall_h
        # if ncall_h < 5:
        #     print(ncall_h, x, x[1] - x[0] ** 2)
        ncall_h += 1
        return x[1] - x[0] ** 2
    nl_h1 = NonlinearConstraint(h1_func, -muFinal-conTol, +muFinal+conTol)

    DBG = False
    if DBG:
        print(nl_h1.fun(xStart * 0.0 + 0.2), objective_function(xStart))
        dummy = 0

    # Run differential evolution with constraints
    start_time = time.perf_counter()
    result = differential_evolution(
        objective_function,
        bounds=bounds,
        constraints=tuple([nl_h1]),
        strategy='best2bin',
        seed=cobraSeed,
        x0=xStart,
        maxiter=500,
    )
    print("time [sec]:", time.perf_counter() - start_time)
    print("Best solution:", result.x)
    print(f"Function value at best solution: {result.fun} with err: {result.fun - (0.75 - muFinal):7.3e}")
    print("Result message:", result.message)
    print("g11_fn(result.x):", g11_fn(result.x))
    print(result.nfev, ncall, ncall_h)
    print(xStart)
    dummy = 0


def run_seq_opt_G11():
    """
    SACOBRA-like solver for COP G11
    """
    start_time = time.perf_counter()
    print(f"Starting run_seq_opt_G11({cobraSeed}) ...")
    verb, verbIter, feval = 1, 10, 500
    muFinal = 1e-4
    G11 = GCOP("G11", mu=muFinal)
    # G11.x0 = np.array([+np.sqrt(0.5-muFinal), 0.5])+0.1   # just to test, if the 2nd solution is found

    cobra = CobraInitializer(G11.x0, G11.fn, G11.name, G11.lower, G11.upper, G11.is_equ,
                             solu=G11.solu, fn_strg=G11.fn_strg, fn_obj=G11.fn_obj,
                             s_opts=SACoptions(verbose=verb, verboseIter=verbIter, feval=feval, cobraSeed=cobraSeed,
                                               ID=IDoptions(initDesign="LHS", initDesPoints=6, rescale=False),
                                               RBF=RBFoptions(degree=2),
                                               EQU=EQUoptions(refinePrint=False, muFinal=muFinal, muType="CONS",
                                                              refine=False, refineAlgo="COBYLA"),
                                               # "L-BFGS-B COBYLA"
                                               # COBYLA is slower, issues warnings, but is a bit more precise
                                               SEQ=SEQoptions(optimizer="DE_OPT", finalEpsXiZero=True,
                                                              DRC_active=False, conTol=conTol,
                                                              trueFuncForSurrogates=True)))  #
    c2 = CobraPhaseII(cobra)
    s_res = c2.cobra.sac_res
    c2.p2.gama = 1e-1
    c2.p2.ro = c2.p2.gama * s_res['l']  # 0.001 #
    # make dummy surrogate models (needed by check_if_cobra_optimizable, even if trueFuncForSurrogates == True):
    c2.p2 = Surrogator.trainSurrogates(c2.cobra, c2.p2)
    # c2.cobra.sac_res['muVec'] = np.array([muFinal, muFinal])  # this is now done via 'muType="CONS"'
    # c2.p2.currentMu = s_res['muVec'][0]                       # this is now done in Phase2Vars.__init__()

    xStart = c2.p2.rs1.random_start(c2.cobra, c2.p2)
    check_if_cobra_optimizable(c2.cobra, c2.p2)
    SeqOptimizer(xStart, c2.cobra, c2.p2)

    result = c2.p2.opt_res
    print("time [sec]:", time.perf_counter() - start_time)
    print("Best solution:", result.x)
    print(f"Function value at best solution: {result.minf} with err: {result.minf - (0.75 - muFinal):7.3e}")
    print("Result message:", result.res_code)
    print("G11.fn(result.x):", G11.fn(result.x))
    print(result.fe_max, G11.ncall)
    print(xStart)

    dummy = 0


if __name__ == '__main__':
    run_demo_G11()
    # Output:
    # --- conTol=0, muFinal=1e-04, maxiter=1000, seed=42: ---
    # time [sec]: 0.42761070001870394
    # Best solution: [-0.70703603  0.49999994]
    # Function value at best solution: 0.7499000102394813 with err: 1.024e-08
    # Result message: Optimization terminated successfully.
    # g11_fn(result.x): array([7.49900010e-01, 9.99897605e-05])      fully feasible
    # 142 142 20092
    #
    run_seq_opt_G11()
    # Output:
    # time [sec]: 0.7707822999800555
    # Best solution: [0.70703603 0.49999994]
    # Function value at best solution: 0.7499000102394817 with err: 1.024e-08
    # Result message: Optimization terminated successfully.
    # G11.fn(result.x): [7.49900010e-01 9.99897605e-05]
    # (148,) 157
