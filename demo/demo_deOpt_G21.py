# Demonstrate that scipy's differential evolution can solve COP G21
# Compare demo solver run_demo_G21() with SACOBRA-like solver run_seq_opt_G21()
#
# Results (printed below): TODO
#
#
import time
import numpy as np
from scipy.optimize import differential_evolution, Bounds, LinearConstraint, NonlinearConstraint

from cobraInit import CobraInitializer
from cobraPhaseII import CobraPhaseII
from gCOP import GCOP
from innerFuncs import PlogSquasher
from opt.equOptions import EQUoptions
from opt.idOptions import IDoptions
from opt.isaOptions import ISAoptions, O_LOGIC
from opt.rbfOptions import RBFoptions
from opt.riOptions import RIoptions
from opt.sacOptions import SACoptions
from opt.seqOptions import SEQoptions
from seqFuncFactory import check_if_cobra_optimizable
from seqOptimizer import SeqOptimizer
from surrogator import Surrogator

conTol = 0.0  # 1e-5  #
muFinal = 1e-4
cobraSeed = 42
ncall, ncall_g, ncall_h = 0, 0, 0


def run_demo_G21():
    """
    Demo solver for COP G21, self-contained
    """
    def g21_fn(x):
        return np.array([+x[0],  # obj
                         -x[0] + 35 * (x[1] ** 0.6) + 35 * (x[2] ** 0.6),  # g1
                         -300 * x[2] + 7500 * x[4] - 7500 * x[5] - 25 * x[3] * x[4] + 25*x[3]*x[5] + x[2]*x[3],  # h1
                         +100 * x[1] + 155.365 * x[3] + 2500 * x[6] - x[1] * x[3] - 25 * x[3] * x[6] - 15536.5,  # h2
                         -x[4] + np.log(-x[3] + 900),  # h3
                         -x[5] + np.log(x[3] + 300),  # h4
                         -x[6] + np.log(-2 * x[3] + 700)  # h5
                         ])

    # Define variable bounds
    G21_lower = [0.0, 0.0, 0.0, 100.0, 6.3, 5.9, 4.5]
    G21_upper = [1000.0, 40.0, 40.0, 300.0, 6.7, 6.4, 6.25]
    G21_solu = np.array([193.724510070034967, 5.56944131553368433e-27, 17.3191887294084914,
                         100.047897801386839, 6.68445185362377892, 5.99168428444264833, 6.21451648886070451
                        ])        # obj = 193.7245100 with maxViol = 1.0e-04
    gamma = 15                                          # CAUTION: values gamma <= 20 lead to ValueError exception
    lower = np.maximum(G21_solu - gamma, G21_lower)     # "array must not contain infs or NaNs"
    upper = np.minimum(G21_solu + gamma, G21_upper)     #
    lower = G21_lower           # this works (gives warning "delta_grad == 0.0" and "max num of iter exceeded", but
    upper = G21_upper           # runs through) and produces good solution with err = 1e-3
    bounds = Bounds(lower, upper)

    # Define a linear constraint: 2*x[0] + x[1] >= 1
    linear_constraint = LinearConstraint([[2, 1]], [1], [np.inf])

    # Define a nonlinear inequality constraint: x[0]**2 + x[1] <= 5
    def g1_func(x):
        global ncall_g
        ncall_g += 1
        return -x[0] + 35 * (np.abs(x[1]) ** 0.6) + 35 * (np.abs(x[2]) ** 0.6)
    nl_g1 = NonlinearConstraint(g1_func, -np.inf, conTol)

    # # define constraint funcs more in the manner of GCOP:
    h_strgs = [
        # "global ncall\n ncall += 1\n lambda x: x[0]",     # obj - does not work
        "lambda x: x[0]",                                   # obj
        "lambda x: -300 * x[2] + 7500 * x[4] - 7500 * x[5] - 25 * x[3] * x[4] + 25 * x[3] * x[5] + x[2] * x[3]",    # h1
        "lambda x: +100 * x[1] + 155.365 * x[3] + 2500 * x[6] - x[1] * x[3] - 25 * x[3] * x[6] - 15536.5",          # h2
        "lambda x: -x[4] + np.log(-x[3] + 900)",            # h3
        "lambda x: -x[5] + np.log(x[3] + 300)",             # h4
        "lambda x: -x[6] + np.log(-2 * x[3] + 700)",        # h5
        ]

    # Define nonlinear equality constraints: -muFinal <= cond(x) <= muFinal
    def h1_func(x):
        global ncall_h
        ncall_h += 1
        return -300 * x[2] + 7500 * x[4] - 7500 * x[5] - 25 * x[3] * x[4] + 25 * x[3] * x[5] + x[2] * x[3]
    # nl_h1 = NonlinearConstraint(lambda x: g21_fn(x)[2], -muFinal-conTol, +muFinal+conTol)
    # nl_h1 = NonlinearConstraint(h1_func, -muFinal-conTol, +muFinal+conTol)
    nl_h1 = NonlinearConstraint(eval(h_strgs[1]), -muFinal-conTol, +muFinal+conTol)

    def h2_func(x):
        return +100 * x[1] + 155.365 * x[3] + 2500 * x[6] - x[1] * x[3] - 25 * x[3] * x[6] - 15536.5
    # nl_h2 = NonlinearConstraint(lambda x: g21_fn(x)[3], -muFinal-conTol, +muFinal+conTol)
    # nl_h2 = NonlinearConstraint(h2_func, -muFinal-conTol, +muFinal+conTol)
    nl_h2 = NonlinearConstraint(eval(h_strgs[2]), -muFinal-conTol, +muFinal+conTol)

    def h3_func(x):
        return -x[4] + np.log(-x[3] + 900)
    # nl_h3 = NonlinearConstraint(lambda x: g21_fn(x)[4], -muFinal-conTol, +muFinal+conTol)
    # nl_h3 = NonlinearConstraint(h3_func, -muFinal-conTol, +muFinal+conTol)
    nl_h3 = NonlinearConstraint(eval(h_strgs[3]), -muFinal-conTol, +muFinal+conTol)

    def h4_func(x):
        return -x[5] + np.log(x[3] + 300)
    # nl_h4 = NonlinearConstraint(lambda x: g21_fn(x)[5], -muFinal-conTol, +muFinal+conTol)
    # nl_h4 = NonlinearConstraint(h4_func, -muFinal-conTol, +muFinal+conTol)
    nl_h4 = NonlinearConstraint(eval(h_strgs[4]), -muFinal-conTol, +muFinal+conTol)

    def h5_func(x):
        return -x[6] + np.log(-2 * x[3] + 700)
    # nl_h5 = NonlinearConstraint(lambda x: g21_fn(x)[6], -muFinal-conTol, +muFinal+conTol)     # 15.15 sec
    # nl_h5 = NonlinearConstraint(h5_func, -muFinal-conTol, +muFinal+conTol)                    #  8.94 sec
    nl_h5 = NonlinearConstraint(eval(h_strgs[5]), -muFinal-conTol, +muFinal+conTol)             #  9.26 sec

    # Define the objective function
    def objective_function(x):
        global ncall
        ncall += 1
        return x[0]
    objective_function2 = eval(h_strgs[0])      # alternative, does not include ncall
    G21_true_opt = objective_function2(G21_solu)

    # # --- does not really work with a function body containing lambda ---
    # def create_func_from_string(name, args, body):
    #     namespace = {}
    #     # Construct the function definition string
    #     func_code = f"def {name}({args}):\n    {body}"
    #     # Execute the code in the isolated namespace
    #     exec(func_code, namespace)
    #     # Retrieve the function object
    #     return namespace[name]
    # # my_func = create_func_from_string("greet", "name", "print(f'Hello, {name}')")
    # # my_func("Alice")  # Output: Hello, Alice
    # # objective_function2 = create_func_from_string("obj", "x", "lambda x: x[0]")

    # Run differential evolution with constraints
    start_time = time.perf_counter()
    xStart = np.array([241.58679755,  18.66783299,   0.95821969, 255.13048001,
                       6.48825972, 5.97023322,   4.8710353])       # copied from run_seq_opt_G21()

    # # --- debug only ---
    # with open('../test/xstart_tst.pkl', 'rb') as input:
    #     xStart = pickle.load(input)             # <-- this leads to crash in DE(!)
    # xStart[1] = 13.7
    # print(xStart)       # [1.93725526e+02 3.72049972e-09 1.73192843e+01 1.00047404e+02
    #                     #  6.68445345e+00 5.99168207e+00 6.21451705e+00]

    assert np.all(lower <= xStart)
    assert np.all(xStart <= upper)
    result = differential_evolution(
        objective_function,
        bounds=bounds,
        constraints=tuple([nl_g1, nl_h1, nl_h2, nl_h3, nl_h4, nl_h5]),
        strategy='best2bin',
        seed=42,
        x0=xStart,
        maxiter=1000,
        tol=1e-6,
    )
    print("time [sec]:", time.perf_counter() - start_time)
    print("Best solution:", result.x)
    print(f"Function value at best solution: {result.fun} with err: {result.fun - G21_true_opt:7.3e}")
    print("Result message:", result.message)
    print(g21_fn(result.x))
    print(result.nfev, ncall, ncall_g, ncall_h)
    print(xStart)


def run_seq_opt_G21():
    """
    SACOBRA-like solver for COP G21
    """
    start_time = time.perf_counter()
    print(f"Starting run_seq_opt_G21({cobraSeed}) ...")
    verb, verbIter, feval = 1, 10, 100
    muFinal = 1e-4
    G21 = GCOP("G21", mu=muFinal)
    dim = G21.dimension
    idp = (dim + 1) * (dim + 2) // 2

    equ = EQUoptions(muGrow=100, muDec=2.6, muType="CONS", muFinal=muFinal,  # 1e-7  # , mu4inequality=True
                     refine=False, refinePrint=False, refineAlgo="L-BFGS-B")  # "L-BFGS-B COBYLA"
    cobra = CobraInitializer(G21.x0, G21.fn, G21.name, G21.lower, G21.upper, G21.is_equ,
                             solu=G21.solu, fn_strg=G21.fn_strg, fn_obj=G21.fn_obj,
                             s_opts=SACoptions(verbose=verb, verboseIter=verbIter, feval=feval, cobraSeed=cobraSeed,
                                               ID=IDoptions(initDesign="LHS", initDesPoints=idp, rescale=False),
                                               RBF=RBFoptions(degree=2),  # kernel="gaussian",
                                               # RBF=RBFoptions(interpolator="sacobra", degree=1.5), # worse
                                               ISA=ISAoptions(onlinePLOG=O_LOGIC.MIDPTS, TGR=np.inf),
                                               # , TFR_DRC=100, TFRange=0
                                               EQU=equ,
                                               RI=RIoptions(repairInfeas=True, eps2=0, q=3, repairMargin=np.inf,
                                                            checkIt=False, trueFuncForSurrogates=True),  #
                                               SEQ=SEQoptions(optimizer="DE_OPT", finalEpsXiZero=True,
                                                              epsilonMax=0.0, epsilonInit=0.0, feMax=1000,
                                                              DRC_active=False, conTol=conTol,
                                                              trueFuncForSurrogates=True)))  #
    c2 = CobraPhaseII(cobra)

    # --- to make it step-by-step similar to cobraPhaseII.py ---
    s_opts = c2.cobra.sac_opts
    s_res = c2.cobra.sac_res
    assert c2.p2.ev1.state == "initialized"
    PlogSquasher.reset_warn_counter()
    # make dummy surrogate models (needed by check_if_cobra_optimizable, even if trueFuncForSurrogates == True):
    c2.p2 = Surrogator.trainSurrogates(c2.cobra, c2.p2)
    # c2.cobra.sac_res['muVec'] = np.array([muFinal, muFinal])  # this is now done via 'muType="CONS"'
    # c2.p2.currentMu = s_res['muVec'][0]                       # this is now done in Phase2Vars.__init__()
    #
    # bug fix 2025-09-16: ensure that EPS is 0.0 if epsilonMax is 0.0:
    c2.p2.EPS = min(c2.p2.EPS, s_opts.SEQ.epsilonMax)

    c2.p2.gama = s_opts.XI[(c2.p2.globalOptCounter % s_opts.XI.size)]
    c2.p2.ro = c2.p2.gama * s_res['l']  # 0.001 #
    # train RBF surrogate models (skipped in case trueFuncForSurrogates == True):
    if not s_opts.SEQ.trueFuncForSurrogates:
        c2.p2 = Surrogator.trainSurrogates(c2.cobra, c2.p2)

    xStart = c2.p2.rs1.random_start(c2.cobra, c2.p2)
    check_if_cobra_optimizable(c2.cobra, c2.p2)

    # # --- debug only ---
    # # this reads in what CobraPhaseII.debug_pickle_dump has written
    # xStart = c2.debug_pickle_load(xStart)

    SeqOptimizer(xStart, c2.cobra, c2.p2)

    G21_true_opt = G21.fn(G21.solu)[0]
    result = c2.p2.opt_res
    print("time [sec]:", time.perf_counter() - start_time)
    orig_x = cobra.rw.inverse(result.x)  # if cobra.sac_opts.ID.rescale else result.x
    print("Best solu COBRA:", result.x)
    print("Best solution:", orig_x)
    print(f"Function value at best solution: {result.minf} with err: {result.minf - G21_true_opt:7.3e}")
    print("Result message:", result.res_code)
    print("G21.fn(result.x):", G21.fn(orig_x))
    print(result.fe_max, G21.ncall)
    orig_xStart = cobra.rw.inverse(xStart)
    print(orig_xStart)


if __name__ == '__main__':
    run_demo_G21()
    # Output:
    # --- lower=G21_lower, upper=G21_upper ---
    # --- conTol=1e-5, maxiter=1000, seed=42: ---
    # time [sec]: 8.18155
    # Best solution: [1.93720233e+02 3.87856349e-09 1.73184930e+01 1.00050550e+02
    #  6.68443958e+00 5.99169986e+00 6.21451438e+00]
    # Function value at best solution: 193.7202334069827  with err: -4.200e-03
    # g21_fn(result.x): array([ 1.93720233e+02, -7.84782789e-05,  2.24442806e-06,  1.08196416e-04,
    #         1.08958091e-04, -1.08947731e-04, -1.08503892e-04])      fully feasible and better obj than G21_solu
    #
    # --- conTol= 0.0, maxiter=1000, seed=42: ---
    # time [sec]: 8.946930899983272
    # Best solution: [1.93725710e+02, 5.37960839e-09, 1.73192837e+01, 1.00047394e+02,
    #  6.68445343e+00, 5.99168208e+00, 6.21451704e+00]
    # Function value at best solution: 193.72552185017398 with err: 1.012e-03
    # g21_fn(result.x): array([ 1.93725710e+02, -1.80739649e-04,  9.03024102e-07,  9.82923484e-05,
    #  9.90548045e-05, -9.90548579e-05, -9.85402219e-05])   fully feasible and slightly worse than G21_solu (err=1.2e-3)
    #
    # --- conTol= 0.0, maxiter=5000, seed=42: ---
    # time [sec]: 49.94771209999453
    # Best solution: [1.93725526e+02 3.71952033e-09 1.73192843e+01 1.00047407e+02
    #  6.68445345e+00 5.99168208e+00 6.21451704e+00]
    # Function value at best solution: 193.72560127122003 with err: 1.091e-03
    # g21_fn(result.x): array([ 1.93725526e+02, -6.84971914e-05,  1.79906920e-06,  9.83173177e-05,
    #         9.90196674e-05, -9.90196310e-05, -9.85906423e-05])      fully feasible and slightly worse than G21_solu

    # G21_solu:      [1.93724510e+02, 5.56944132e-27, 1.73191887e+01, 1.00047898e+02,
    #  6.68445185e+00, 5.99168428e+00, 6.21451649e+00]
    # g21_fn(G21_solu): array([ 1.93724510e+02, -2.84217094e-14,  9.99999975e-05,  1.00000001e-04,
    #         1.00000000e-04, -1.00000000e-04, -1.00000000e-04])

    run_seq_opt_G21()
    # --- result with ID.rescale=False (and new constr interface g_constr2, h_constr2): ---
    # time [sec]: 9.656030699959956
    # Best solu COBRA: [1.93725526e+02 3.72049972e-09 1.73192843e+01 1.00047404e+02
    #  6.68445345e+00 5.99168207e+00 6.21451705e+00]
    # Best solution: [1.93725526e+02 3.72049972e-09 1.73192843e+01 1.00047404e+02
    #  6.68445345e+00 5.99168207e+00 6.21451705e+00]
    # Function value at best solution: 193.72552635045557 with err: 1.016e-03
    # Result message: Maximum number of iterations has been exceeded.
    # G21.fn(result.x): [ 1.93725526e+02 -6.82559421e-05  1.81038172e-06  9.83007940e-05
    #   9.90240140e-05 -9.90239551e-05 -9.85878020e-05]
    # (2936,) 43
    # --- earlier result with ID.rescale=True (and old constr interface g_constr, h_const): ---
    # time [sec]: 40.39621179999085
    # Best solu COBRA: [-0.61254564 -1.  -0.1340358  -0.99952254  0.92226873 -0.63327045  0.95944804]
    # Best solution: [1.93727180e+02 5.49912671e-08 1.73192839e+01 1.00047746e+02
    #  6.68445375e+00 5.99168239e+00 6.21451703e+00]
    # Function value at best solution: 193.72717977426686 with err: 2.670e-03
    # Result message: Maximum number of iterations has been exceeded.
    # G21.fn(result.x): [ 1.93727180e+02 -4.88231582e-04  2.22877588e-07  9.90332355e-05
    #   9.82979649e-05 -9.84820275e-05 -9.99368257e-05]
    # (4984,) 1295585