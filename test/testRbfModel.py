import unittest
import numpy as np
from cobraInit import CobraInitializer
from rbfModel import RBFmodel
from opt.idOptions import IDoptions
from opt.sacOptions import SACoptions
from scipy.interpolate import RBFInterpolator

verb = 1


def fn_lin(x):
    return x[:, 0] * 2 + x[:, 1] * 3


class TestRbfModel(unittest.TestCase):
    """
        Several tests for component :class:`RBFmodel`
    """

    ### Don't use __init__! This leads to problems for __main__ in testRbfsacob.py when it calls
    ###          trm = testRbfModel()
    ### in line 51. Better set self.val=24 (or trm.val=24) explicitely in each method that calls my_rng2.
    # def __init__(self):
    #     super().__init__()
    #     self.val = 24

    def test_rbf_model(self):
        """
            Generate an interpolating cubic RBF model for ``fn_rbf``: :math:`\\mathbb{R}^2 \\rightarrow \\mathbb{R}^2`
            from ``nobs=500`` observations (training points). Test whether the model values at ``ngrid*ngrid`` points
            (usually different to the training points) are close to the values of ``fn_rbf`` itself.

            Result: The relative errors are in all cases < 1e-4.
        """
        nobs = 500
        ngrid = 22  # number of grid points per dimension in xgrid

        def fn_rbf(x):
            return np.array([3 * np.sum(x ** 2), np.sum(x) - 1])
        is_equ = np.array([False])

        print(f"\n[test_rbf_model started with nobs = {nobs}, ngrid = {ngrid}]")
        x0 = np.array([2.5, 2.4])
        d = x0.size
        lower = np.array([-1, -1])
        upper = np.array([ 1,  1])
        cobra = CobraInitializer(x0, fn_rbf, "fName", lower, upper, is_equ,
                                 s_opts=SACoptions(verbose=verb, feval=2*nobs, cobraSeed=44,
                                                   ID=IDoptions(initDesign="RAND_REP",
                                                                initDesPoints=nobs)))
        sac_res = cobra.get_sac_res()
        xobs = sac_res['A']
        # observations for one output model, provided as vector
        # yobs = sac_res['Fres']
        # observations for two output models, arranged in columns [i.e. yobs.shape = (100,2)]:
        yobs = np.hstack((sac_res['Fres'].reshape(-1, 1), sac_res['Gres']))
        rbf_model = RBFmodel(xobs, yobs, kernel="cubic", degree=None)

        xr = 0.8    # test only the inner region of input space [-1,1]^2
                    # (because the approximation error is too large at the outer rim)
        xgrid = np.mgrid[-xr:xr:ngrid * 1j, -xr:xr:ngrid * 1j]
        xflat = xgrid.reshape(d, -1).T
        yflat = rbf_model(xflat)

        ### this test is nonsense, since yobs and yo2 are exactly the same because rbf_model is interpolating (!):
        # yo2 = np.apply_along_axis(fn_rbf, axis=1, arr=xobs)
        # assert np.allclose(yobs, yo2)

        ### this test is sensible, because other points than the training points (xobs,yobs) are tested:
        yf2 = np.apply_along_axis(fn_rbf, axis=1, arr=xflat)
        ydelta = np.abs((yflat-yf2)/yflat)
        print("max. relative deviation =", np.max(ydelta))
        assert np.allclose(yflat, yf2, rtol=1e-3)
        print("[test_rbf_model passed]")

    def test_linear_func(self):
        """
            Generate an interpolating cubic RBF model for the  linear function ``fn_lin``:
            :math:`\\mathbb{R} \\rightarrow \\mathbb{R}` in different forms: 10/100 observations, with/without
            polynomial tail. Test whether ``fn_lin`` is RBF-modeled the same way in Python and in R (see
            ``demo-rbf3.R``). Uses RNG ``self.my_rng2`` that avoids cycles.

            Result: The absolute errors are in all cases < 1e-7.
        """
        d = 2
        ngrid = 100
        for nobs in [10, 100]:
            for deg in [-1, 1]:         # [without, with] polynomial tail
                self.val = 24  # setting for my_rng2
                print(f"\n[test_linear_func started with d = {d}, nobs = {nobs}, deg = {deg}]")
                xobs = self.my_rng2(nobs, d)  # better: avoids cycles
                yobs = fn_lin(xobs)
                xgrid = np.mgrid[-1:1:ngrid * 1j, -1:1:ngrid * 1j]
                xflat = xgrid.reshape(d, -1).T
                # print(xflat.shape)  # (ngrid**d, d)
                rbf_model = RBFInterpolator(xobs, yobs, kernel="cubic", degree=deg, smoothing=0)
                yflat = rbf_model(xflat)
                ygrid = yflat.reshape(ngrid, ngrid)

                ytrue = fn_lin(xflat).reshape(ngrid, ngrid)
                delta = ytrue - ygrid
                print(f"avg abs |ytrue - ygrid| = {np.mean(np.abs(delta)):.4e}")

                L = ngrid // 2
                ytest = ygrid[50:60, L - 1]

                np.set_printoptions(7)
                print(ytest)

                # Test numerical equivalence with R-implementation.
                # These are the values generated with demo-rbf3.R and my_rng2 (avoid cycles!) on the R-side:
                if nobs == 10:
                    if deg == -1:   # ptail=F in R
                        ytest_from_R = np.array([ 0.0354658,  0.0771876,  0.1187900,  0.1602633,  0.2015979,
                                                  0.2427840,  0.2838119,  0.3246720,  0.3653547,  0.4058505])
                    elif deg == 1:  # ptail=T in R
                        ytest_from_R = np.array([-0.0101010,  0.0303030,  0.0707071,  0.1111111,  0.1515152,
                                                  0.1919192,  0.2323232,  0.2727273,  0.3131313,  0.3535354])
                    else:
                        raise NotImplementedError(f"No data from the R side for nobs={nobs} and deg={deg}")
                elif nobs == 100:
                    if deg == -1:   # ptail=F in R
                        ytest_from_R = np.array([-0.0100817,  0.0303251,  0.0707322,  0.1111393,  0.1515466,
                                                  0.1919538,  0.2323607,  0.2727671,  0.3131729,  0.3535776])
                    elif deg == 1:  # ptail=T in R
                        ytest_from_R = np.array([-0.0101010,  0.0303030,  0.0707071,  0.1111111,  0.1515152,
                                                  0.1919192,  0.2323232,  0.2727273,  0.3131313,  0.3535354])
                    else:
                        raise NotImplementedError(f"No data from the R side for nobs={nobs} and deg={deg}")
                else:
                    raise NotImplementedError(f"No data from the R side for nobs={nobs}")

                delta = ytest - ytest_from_R
                print(f"max |ytest - ytest_from_R| = {np.max(np.abs(delta)):.4e}")
                assert np.allclose(ytest, ytest_from_R, rtol=1e-4)
                print(f"[test_linear_func with nobs={nobs}, deg={deg} passed]")

    def my_rng2(self, n, d) -> np.ndarray:
        """
        A very simple RNG to create reproducible random numbers in Python and R.

        CAUTION: Is intended for small samples (n<=100) only. For n>100, *cycles* might occur! This can lead
        to singular matrices (and crashes) in RBF interpolation.

        :param n:
        :param d:
        :return:       (n,d)-matrix with random numbers in range [-1, 1[
        """
        MOD = 10**5+7
        OFS = 10 ** 5 - 7
        x = np.zeros((n, d), dtype=np.float32)
        for n_ in range(n):
            for d_ in range(d):
                self.val = (self.val * self.val * np.sqrt(self.val) + OFS) % MOD  # avoid cycles (!)
                x[n_, d_] = 2*self.val/MOD - 1    # map val to float range [-1,1[
        return x


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
    x = np.zeros((n, d), dtype=np.float32)
    for n_ in range(n):
        for d_ in range(d):
            val = (val*val) % MOD           # CAUTION: This might create cycles for larger n (!)
            x[n_, d_] = 2*val/MOD - 1    # map val to float range [-1,1[
    return x


if __name__ == '__main__':
    unittest.main()
