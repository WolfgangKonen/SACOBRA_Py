## The optimization problem is the Hock-Schittkowski problem no. 100
## The optimum value of the objective function should be 680.6300573
## A suitable parameter vector is roughly
##    (2.330, 1.9514, -0.4775, 4.3657, -0.6245, 1.0381, 1.5942)
##
import nlopt
import numpy as np

true_opt = np.array([2.330, 1.9514, -0.4775, 4.3657, -0.6245, 1.0381, 1.5942])
# xstart = true_opt +0.1
xstart =  [1, 2, 0, 4, 0, 1, 1]


def myfunc(x, grad):
    # if grad.size > 0:
    #     grad[0] = 2*(x[0]-10)
    #     grad[1] =10*(x[1]-12)
    #     grad[2] = 4 * x[2]**3
    #     grad[3] = 6 * (x[3]-11)
    #     grad[4] = 60 * x[4]**5
    #     grad[5] = 14 * x[5] - 4*x[6] - 10
    #     grad[6] = 4 * x[6]**3 - 4*x[5] - 8
    val = (x[0] - 10) ** 2 + 5 * (x[1] - 12) ** 2 + x[2] ** 4
    val += 3 * (x[3] - 11) ** 2 + 10 * x[4] ** 6 + 7 * x[5] ** 2
    val += x[6] ** 4 - 4 * x[5] * x[6] - 10 * x[5] - 8 * x[6]
    return val

def myconstraint1(x,grad):
    return +(2 * x[0] ** 2 + 3 * x[1] ** 4 + x[2] + 4 * x[3] ** 2 + 5 * x[4] - 127)
def myconstraint2(x,grad):
    return +(7 * x[0] + 3 * x[1] + 10 * x[2] ** 2 + x[3] - x[4] - 282)
def myconstraint3(x, grad):
    return +(23 * x[0] + x[1] ** 2 + 6 * x[5] ** 2 - 8 * x[6] - 196)
def myconstraint4(x, grad):
    return +(4 * x[0] ** 2 + x[1] ** 2 - 3 * x[0] * x[1] + 2 * x[2] ** 2 + 5 * x[5] - 11 * x[6])


def g_vec_c(result, x, grad):
    result[0] = myconstraint1(x, grad)
    result[1] = myconstraint2(x, grad)
    result[2] = myconstraint3(x, grad)
    result[3] = myconstraint4(x, grad)


# just a check : If a dict or a class object derived from dict is a calling parameter of a function
# and if it is changed inside, then it is also changed for the caller of the function
class SACopts(dict):
   def __init__(self):
       super().__init__()
       self['a'] = 1
       self['c'] = 2
   def myfunc(self, x, grad):
       val = (x[0] - 10) ** 2 + 5 * (x[1] - 12) ** 2 + x[2] ** 4
       val += 3 * (x[3] - 11) ** 2 + 10 * x[4] ** 6 + 7 * x[5] ** 2
       val += x[6] ** 4 - 4 * x[5] * x[6] - 10 * x[5] - 8 * x[6]
       return val

# def my_func(sac_opts):
#     sac_opts['w'] = 2

# sac_opts = SACopts()
# print(sac_opts)
# my_func(sac_opts)
# print(sac_opts)


opt = nlopt.opt(nlopt.LN_COBYLA, 7)
# opt = nlopt.opt(nlopt.LD_MMA, 7)       # does not complete, if no gradient is specified
opt.set_lower_bounds(np.repeat(-5,7))
opt.set_upper_bounds(np.repeat(+5,7))
#opt.set_min_objective(myfunc)
opt.set_min_objective(SACopts().myfunc)     # just a check: we can also pass methods of classes as function? - yes
tol = 0 # 1e-8 #
opt.add_inequality_constraint(myconstraint1, tol)
opt.add_inequality_constraint(myconstraint2, tol)
opt.add_inequality_constraint(myconstraint3, tol)
opt.add_inequality_constraint(myconstraint4, tol)
opt.set_xtol_rel(1e-4)
x = opt.optimize(xstart)
minf = opt.last_optimum_value()
optim = 680.6300573
print(opt.get_algorithm_name())
np.set_printoptions(precision=4)
print(f"optimum  at  ({ x})")
print(f"true optimum ({true_opt})")
print(f"minimum value = {minf}, error = {np.abs(minf-optim):5.4e}")
print(f"true min.val. = {optim}")
print("result code = ", opt.last_optimize_result())

# test that we reach the same results with opt.add_inequality_mconstraint and g_vec_c:
opt.remove_inequality_constraints()
tol = np.repeat(0, 4)
opt.add_inequality_mconstraint(g_vec_c, tol)
x = opt.optimize(xstart)
minf = opt.last_optimum_value()
print(f"optimum  at  ({ x})")
print(f"true optimum ({true_opt})")
print(f"minimum value = {minf}, error = {np.abs(minf-optim):5.4e}")
print(f"true min.val. = {optim}")
print("result code = ", opt.last_optimize_result())

# # just an example to show the usage of **kwargs:
# def greet_me(**kwargs):
#     for key, value in kwargs.items():
#         print("{0} = {1}".format(key, value))
# greet_me(name="yasoob", nam2="esau")

