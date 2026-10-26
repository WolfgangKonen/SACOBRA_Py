# Demonstrate that scipy's differential evolution can solve COP G21
#
import time
import numpy as np
from scipy.optimize import differential_evolution, Bounds, LinearConstraint, NonlinearConstraint

conTol  =  0.0 # 1e-5  #
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


# Define the objective function
def objective_function(x):
    return x[0]


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
    return -x[0] + 35 * (x[1] ** (0.6)) + 35 * (x[2] ** 0.6)


nl_g1 = NonlinearConstraint(g1_func, -np.inf, conTol)


# Define nonlinear equality constraints: -muFinal <= cond(x) <= muFinal
def h1_func(x):
    return -300 * x[2] + 7500 * x[4] - 7500 * x[5] - 25 * x[3] * x[4] + 25 * x[3] * x[5] + x[2] * x[3]
nl_h1 = NonlinearConstraint(h1_func, -muFinal-conTol, +muFinal+conTol)

def h2_func(x):
    return +100 * x[1] + 155.365 * x[3] + 2500 * x[6] - x[1] * x[3] - 25 * x[3] * x[6] - 15536.5
nl_h2 = NonlinearConstraint(h2_func, -muFinal-conTol, +muFinal+conTol)

def h3_func(x):
    return -x[4] + np.log(-x[3] + 900)
nl_h3 = NonlinearConstraint(h3_func, -muFinal-conTol, +muFinal+conTol)

def h4_func(x):
    return -x[5] + np.log(x[3] + 300)
nl_h4 = NonlinearConstraint(h4_func, -muFinal-conTol, +muFinal+conTol)

def h5_func(x):
    return -x[6] + np.log(-2 * x[3] + 700)
nl_h5 = NonlinearConstraint(h5_func, -muFinal-conTol, +muFinal+conTol)

# Run differential evolution with constraints
start_time = time.perf_counter()
result = differential_evolution(
    objective_function,
    bounds=bounds,
    constraints=[nl_g1, nl_h1, nl_h2, nl_h3, nl_h4, nl_h5],
    strategy='best2bin',
    seed=42,
    maxiter=1000,
)
print("time [sec]:", time.perf_counter() - start_time)
print("Best solution:", result.x)
print(f"Function value at best solution: {result.fun} with err: {result.fun - 193.7245100:7.3e}")
print("Result message:", result.message)
print(g21_fn(result.x))
dummy = 0
# Output:
# --- lower=G21_lower, upper=G21_upper ---
# --- conTol=1e-5, maxiter=1000, seed=42: ---
# time [sec]: 8.18155
# Best solution: [1.93720233e+02 3.87856349e-09 1.73184930e+01 1.00050550e+02
#  6.68443958e+00 5.99169986e+00 6.21451438e+00]
# Function value at best solution: 193.7202334069827  with err: -4.200e-03
# g21_fn(result.x): array([ 1.93720233e+02, -7.84782789e-05,  2.24442806e-06,  1.08196416e-04,
#         1.08958091e-04, -1.08947731e-04, -1.08503892e-04])      fully feasible and better objective (!!) than G21_solu
#
# --- conTol= 0.0, maxiter=1000, seed=42: ---
# time [sec]: 8.946930899983272
# Best solution: [1.93725710e+02, 5.37960839e-09, 1.73192837e+01, 1.00047394e+02,
#  6.68445343e+00, 5.99168208e+00, 6.21451704e+00]
# Function value at best solution: 193.72552185017398 with err: 1.012e-03
# g21_fn(result.x): array([ 1.93725710e+02, -1.80739649e-04,  9.03024102e-07,  9.82923484e-05,
#         9.90548045e-05, -9.90548579e-05, -9.85402219e-05])      fully feasible and slightly worse than G21_solu (err=1.2e-3)
#
# --- conTol= 0.0, maxiter=5000, seed=42: ---
# time [sec]: 49.94771209999453
# Best solution: [1.93725526e+02 3.71952033e-09 1.73192843e+01 1.00047407e+02
#  6.68445345e+00 5.99168208e+00 6.21451704e+00]
# Function value at best solution: 193.72560127122003 with err: 1.091e-03
# g21_fn(result.x): array([ 1.93725526e+02, -6.84971914e-05,  1.79906920e-06,  9.83173177e-05,
#         9.90196674e-05, -9.90196310e-05, -9.85906423e-05])      fully feasible and slightly worse than G21_solu (err=1.0e-3)

# G21_solu:      [1.93724510e+02, 5.56944132e-27, 1.73191887e+01, 1.00047898e+02,
#  6.68445185e+00, 5.99168428e+00, 6.21451649e+00]
# g21_fn(G21_solu): array([ 1.93724510e+02, -2.84217094e-14,  9.99999975e-05,  1.00000001e-04,
#         1.00000000e-04, -1.00000000e-04, -1.00000000e-04])
