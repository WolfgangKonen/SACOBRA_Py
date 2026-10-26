# Example from https://pythonguides.com/scipy-differential-evolution/
#
from scipy.optimize import differential_evolution
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint

# Define the objective function
def objective_function(x):
    return x[0]**2 + x[1]**2

# Define variable bounds
bounds = Bounds([0, 0], [10, 10])

# Define a linear constraint: 2*x[0] + x[1] >= 1
linear_constraint = LinearConstraint([[2, 1]], [1], [np.inf])

# Define a nonlinear constraint: x[0]**2 + x[1] <= 5
def nonlinear_constraint_func(x):
    return x[0]**2 + x[1]

nonlinear_constraint = NonlinearConstraint(nonlinear_constraint_func, -np.inf, 5)

# Run differential evolution with constraints
result = differential_evolution(
    objective_function,
    bounds=bounds,
    constraints=(linear_constraint, nonlinear_constraint)
)

print("Best solution:", result.x)
print("Function value at best solution:", result.fun)
# Output:
# Best solution: [0.40000183 0.20001235]
# Function value at best solution: 0.20000640108586462
# Function value at best solution: 0.20000640108583126 (slightly different on https://pythonguides.com/scipy-differential-evolution/)
