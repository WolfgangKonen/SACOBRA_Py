"""
Example from the RPFInterpolator doc page
https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html#scipy.interpolate.RBFInterpolator
with modifications WK:
    - use reproducible random observations (reproducible also on the R side)
    - configurable dimension d=1 or d=2
    - configurable number of observations nobs
    - configurable grid size ngrid
    - cubic RBFs
    - time measurement
    - compute and plot delta to true y values
    - test on numerical equivalence to R-implementation
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator

nobs = 100      # number of observations fed into RBFInterpolator
ngrid = 100     # number of grid points per dimension in xgrid
runs = 10


def my_rng(n, d, seed) -> np.ndarray:
    """
    A very simple RNG to create reproducible random numbers in Python and R.

    CAUTION: Is intended for small samples (n<=100) only. For n>100, *cycles* might occur! This can lead
    to singular matrices (and crashes) in RBF interpolation.

    :param n:
    :param d:
    :param seed:
    :return:       (n,d)-matrix with random numbers in range [-1, 1[
    """
    MOD = 10**5+7
    val = seed
    x = np.zeros((n,d), dtype=np.float32)
    for n_ in range(n):
        for d_ in range(d):
            val = (val*val) % MOD           # CAUTION: This might create cycles for larger n (!)
            x[n_,d_] = 2*val/MOD - 1    # map val to float range [-1,1[
    return x

def fn(x):
    return np.sum(x, axis=1)*np.exp(-6*np.sum(x**2, axis=1))

def interp_func(d, nobs, ngrid, runs):
    print(f"\n*** [interp_func] starting with d = {d} ... ***")
    rng = np.random.default_rng()
    # xobs = 2*rng.random((nobs,d))-1
    xobs = my_rng(nobs, d,24)      # CAUTION: has cycles for n>100, but needed for numerical-equivalence check
    yobs = fn(xobs)
    print(yobs[0:9])
    if d == 1:
        xgrid = np.mgrid[-1:1:ngrid * 1j]
    else:       # i.e. d == 2
        xgrid = np.mgrid[-1:1:ngrid * 1j, -1:1:ngrid * 1j]
    xflat = xgrid.reshape(d, -1).T
    print(xflat.shape)              # (ngrid**d, d)
    start = time.perf_counter()
    for r in range(runs):
        # s = 0.005 if nobs>100 else 0
        ### The above line with s=0.005 was only necessary with wrong (cyclic) xobs from my_rng for nobs>100,
        ### which would lead to a singular matrix when smoothing=0. With correct xobs from rng (no cyclic data), all
        ### works fine with s=0 even for nobs=1000.
        s = 0
        rbf_model = RBFInterpolator(xobs, yobs, kernel="cubic", degree=None, smoothing=s)     # cubic kernel with linear polynomial tail
        yflat = rbf_model(xflat)
        ### This can be also done in one line
        ###     yflat = RBFInterpolator(xobs, yobs, kernel="cubic")(xflat)
        ### but we want to be able to inspect rbf_model in debugger
    print(f"avg time RBFInterpolator {1000*(time.perf_counter()-start)/runs:.4f} msec (avg from {runs} runs)")
    if d == 1:
        ygrid = yflat
    else:       # i.e. d == 2
        ygrid = yflat.reshape(ngrid, ngrid)

        # plot a contour-like color mesh with the observations overlaid as circles:
        fig, ax = plt.subplots()
        ax.pcolormesh(*xgrid, ygrid, vmin=-0.25, vmax=0.25, shading='gouraud')
        p = ax.scatter(*xobs.T, c=yobs, s=50, ec='k', vmin=-0.25, vmax=0.25)
        fig.colorbar(p)
        plt.show()

    ytrue =fn(xflat)
    if d == 2:
        ytrue = ytrue.reshape(ngrid, ngrid)
    delta = ytrue - ygrid
    print(f"avg abs |ytrue - ygrid| = {np.mean(np.abs(delta)):.4e}")

    # plot for one line: ygrid, ytrue, delta
    if d == 1:
        plt.plot(xgrid, ygrid, 'bo', label=f'ygrid')
        plt.plot(xgrid, ytrue, 'r-', label=f'ytrue')
        plt.plot(xgrid, delta, 'g-', label=f'delta')
        ytest = ygrid[50:60]
    else:       # i.e. d == 2
        L = ngrid//2
        plt.plot(xgrid[0,:,L], ygrid[:,L], 'bo', label=f'ygrid[:,{L}]')
        plt.plot(xgrid[0,:,L], ytrue[:,L], 'r-', label=f'ytrue[:,{L}]')
        plt.plot(xgrid[0,:,L], delta[:,L], 'g-', label=f'delta[:,{L}]')
        ytest = ygrid[50:60,L-1]
    plt.title('One line')
    plt.xlabel('xgrid ',fontsize=16)
    plt.ylabel('y',fontsize=16)
    #plt.ylim(top=0.5,bottom=0.0)
    plt.legend()
    plt.show()

    np.set_printoptions(7)
    print(ytest)
    if d == 2:
        # Test numerical equivalence with R-implementation.
        # These are the values generated with demo-rbf2-R.R on the R-side:
        if nobs == 10:
            ytest_from_R = np.array([-0.0958555, -0.0950409, -0.0940627, -0.0929260, -0.0916367,
                                     -0.0902010, -0.0886258, -0.0869181, -0.0850855, -0.0831358])
        elif nobs == 100:
            ytest_from_R = np.array([-0.0097614,  0.0088733,  0.0274552,  0.0457256,  0.0634270,
                                      0.0803093,  0.0961359,  0.1106909,  0.1237856,  0.1352644])
        else:
            raise NotImplementedError(f"No data from the R side for nobs={nobs}")

        delta = ytest - ytest_from_R
        print(f"avg |ytest - ytest_from_R| = {np.mean(np.abs(delta)):.4e}")
        assert np.allclose(ytest, ytest_from_R)

# print(my_rng(2,3,24))


# interp_func(1, nobs, ngrid, runs)
interp_func(2, nobs, ngrid, runs)
