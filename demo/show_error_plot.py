import numpy as np
import matplotlib.pyplot as plt
from cobraInit import CobraInitializer
from gCOP import GCOP

def show_error_plot(cobra: CobraInitializer, gfunc: GCOP, file=None):
    err = cobra.sac_res['fbestArray'] - gfunc.fbest
    plt.plot(range(err.size), np.abs(err), 'r-', label='error')
    plt.title(gfunc.name, fontsize=20)
    plt.xlabel('func evals ', fontsize=16)
    plt.ylabel('error', fontsize=16)
    plt.subplot(111).set_yscale("log")
    if file is not None:
        plt.savefig(file)
    plt.show()

