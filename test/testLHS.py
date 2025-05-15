import lhsmdu
import matplotlib.pyplot as plt
import numpy as np


def plot_lhs():
    l = lhsmdu.createRandomStandardUniformMatrix(2, 20)  # Monte Carlo sampling
    k = lhsmdu.sample(2, 20)  # Latin Hypercube Sampling with multi-dimensional uniformity
    k = np.array(k)
    l = np.array(l)
    fig = plt.figure()
    ax = fig.gca()
    ax.set_xticks(np.arange(0, 1, 0.1))
    ax.set_yticks(np.arange(0, 1, 0.1))
    plt.scatter(k[0], k[1], color="g", label="LHS-MDU")
    plt.scatter(l[0], l[1], color="r", label="MC")
    plt.grid()
    plt.show()
    dummy = 0


k = lhsmdu.sample(2, 20)  # Latin Hypercube Sampling with multi-dimensional uniformity
#print(k)
plot_lhs()
