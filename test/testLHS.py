import lhsmdu
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.qmc import LatinHypercube


def plot_lhs(k, n, d):
    l = lhsmdu.createRandomStandardUniformMatrix(d, n)  # Monte Carlo sampling
    l = np.array(l)
    fig = plt.figure()
    ax = fig.gca()
    ax.set_xticks(np.arange(0, 1, 0.1))
    ax.set_yticks(np.arange(0, 1, 0.1))
    plt.scatter(k[0], k[1], color="g", label="LHSU")
    plt.scatter(l[0], l[1], color="r", label="MC")
    plt.grid()
    plt.legend()
    plt.show()


d = 2
n = 80
seed = 42+1

SCIPY_LHS = True
if SCIPY_LHS:
    engine = LatinHypercube(d=d, rng=seed)
    sam = engine.random(n=n)    #  shape=(n,d), uniform random in [0,1)
    k = np.array(sam).T
else:  # use package lhsmdu --- deprecated
    k = np.array(lhsmdu.sample(d, n, randomSeed=seed))  # Latin Hypercube Sampling with multi-dimensional uniformity

#print(k)
plot_lhs(k, n, d)

