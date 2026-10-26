## The optimization problem is the Hock-Schittkowski problem no. 100
## The optimum value of the objective function should be 680.6300573
## A suitable parameter vector is roughly
##    (2.330, 1.9514, -0.4775, 4.3657, -0.6245, 1.0381, 1.5942)
##
import nlopt
import numpy as np

conTol  = 0.0 # 1e-5
muFinal = 1e-4


def g21_fn(x):
    return np.array([+x[0],  # obj
                     -x[0] + 35 * (x[1] ** (0.6)) + 35 * (x[2] ** 0.6),  # g1
                     -300 * x[2] + 7500 * x[4] - 7500 * x[5] - 25 * x[3] * x[4] + 25 * x[3] * x[5] + x[2] * x[3],  # h1
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
true_opt = G21_solu
optim = g21_fn(G21_solu)[0]

xstart =  np.minimum(G21_solu + 5, G21_upper)
gamma = 15                                          # CAUTION: values gamma <= 20 lead to ValueError exception
lower = np.maximum(G21_solu - gamma, G21_lower)     # "array must not contain infs or NaNs"
upper = np.minimum(G21_solu + gamma, G21_upper)     #
lower = G21_lower           # this works (gives warning "delta_grad == 0.0" and "max num of iter exceeded", but
upper = G21_upper           # runs through) and produces good solution with err = 1e-3


# Define the objective function
def objective_function(x, grad):
    return x[0]

def g1_func(x, grad):
    return +(-x[0] + 35 * (x[1] ** (0.6)) + 35 * (x[2] ** 0.6))

def h1_func(x, grad):
    return -300 * x[2] + 7500 * x[4] - 7500 * x[5] - 25 * x[3] * x[4] + 25 * x[3] * x[5] + x[2] * x[3]

def h2_func(x, grad):
    return +100 * x[1] + 155.365 * x[3] + 2500 * x[6] - x[1] * x[3] - 25 * x[3] * x[6] - 15536.5

def h3_func(x, grad):
    return -x[4] + np.log(-x[3] + 900)

def h4_func(x, grad):
    return -x[5] + np.log(x[3] + 300)



def h_vec_c(result, x, grad):
    result[0] = -h1_func(x, grad)
    result[1] = -h2_func(x, grad)
    result[2] = -h3_func(x, grad)
    result[3] = -h4_func(x, grad)
    result[4] = +h4_func(x, grad)
    # result[4] = -h1_func(x, grad) - muFinal
    # result[5] = -h2_func(x, grad) - muFinal
    # result[6] = -h3_func(x, grad) - muFinal
    # result[7] = -h4_func(x, grad) - muFinal


opt = nlopt.opt(nlopt.LN_COBYLA, 7)
opt.set_lower_bounds(lower)
opt.set_upper_bounds(upper)
opt.set_min_objective(objective_function)
opt.set_xtol_rel(1e-6)
opt.set_maxeval(10000)
tol = np.repeat(muFinal, 5)
opt.add_inequality_constraint(g1_func, conTol)
opt.add_equality_mconstraint(h_vec_c, tol)
try:
    x = opt.optimize(xstart)
except nlopt.RoundoffLimited:
    print(f"WARNING: nlopt.RoundoffLimited exception "
          f"(result code {opt.last_optimize_result()})")
    x = xstart.copy()

minf = opt.last_optimum_value()
print(f"optimum  at  ({ x})")
print(f"true optimum ({true_opt})")
print(f"minimum value = {minf}, error = {minf-optim:5.4e}")
print(f"true min.val. = {optim}")
print("result code = ", opt.last_optimize_result())
print(g21_fn(x))

