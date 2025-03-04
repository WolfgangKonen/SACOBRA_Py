import nlopt
import numpy as np


def myfunc(x, grad):
    if grad.size > 0:
        grad[0] = 0.0
        grad[1] = 0.5 / np.sqrt(x[1])
    return np.sqrt(x[1])

def myconstraint(x, grad, a, b):
    if grad.size > 0:
        grad[0] = 3 * a * (a*x[0] + b)**2
        grad[1] = -1.0
    return (a*x[0] + b)**3 - x[1]


opt = nlopt.opt(nlopt.LN_COBYLA, 2)
# opt = nlopt.opt(nlopt.LD_MMA, 2)
opt.set_lower_bounds([-float('inf'), 0])
opt.set_min_objective(myfunc)
tol = 0 # 1e-8 #
opt.add_inequality_constraint(lambda x,grad: myconstraint(x,grad,2,0), tol)
opt.add_inequality_constraint(lambda x,grad: myconstraint(x,grad,-1,1), tol)
opt.set_xtol_rel(1e-4)
x = opt.optimize([1.234, 5.678])
minf = opt.last_optimum_value()
optim = np.sqrt(8/27)
print(opt.get_algorithm_name())
print(f"optimum  at  ({ x[0]:.9f}, {x[1]:.9f})")
print(f"true optimum ({1/3:.9f}, {8/27:.9f})")
print(f"minimum value = {minf}, error = {np.abs(minf-optim):5.4e}")
print(f"true min.val. = {optim}")
print("result code = ", opt.last_optimize_result())