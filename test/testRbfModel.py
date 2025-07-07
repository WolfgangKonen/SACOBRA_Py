import unittest
import numpy as np
from cobraInit import CobraInitializer
from cobraPhaseII import CobraPhaseII
from rbfModel import RBFmodel
from opt.idOptions import IDoptions
from opt.sacOptions import SACoptions
from scipy.interpolate import RBFInterpolator

verb = 1


def fn(x):
    return x[:, 0] * 2 + x[:, 1] * 3


class TestRbfModel(unittest.TestCase):
    def __init__(self):
        super().__init__()
        self.val = 24

    def test_rbf_model(self):
        nobs = 100
        ngrid = 100  # number of grid points per dimension in xgrid

        def fn_rbf(x):
            return np.array([3 * np.sum(x ** 2), np.sum(x) - 1])
        is_equ=np.array([False])

        x0 = np.array([2.5, 2.4])
        d = x0.size
        lower = np.array([-1, -1])
        upper = np.array([ 1,  1])
        cobra = CobraInitializer(x0, fn_rbf, "fName", lower, upper, is_equ,
                                 s_opts=SACoptions(verbose=verb, feval=2*nobs,
                                                   ID=IDoptions(initDesign="RAND_REP",
                                                                initDesPoints=nobs)))
        sac_res = cobra.get_sac_res()
        xobs = sac_res['A']
        # observations for one output model, provided as vector
        yobs = sac_res['Fres']
        # observations for two output models, arranged in columns:
        yobs = np.hstack((sac_res['Fres'].reshape(-1,1),sac_res['Gres']))
        rbf_model = RBFmodel(xobs, yobs, kernel="cubic", degree=None)

        xgrid = np.mgrid[-1:1:ngrid * 1j, -1:1:ngrid * 1j]
        xflat = xgrid.reshape(d, -1).T
        yflat = rbf_model(xflat)

        yo2 = np.apply_along_axis(fn_rbf, axis=1, arr=xobs)
        assert np.allclose(yobs, yo2)
        print("[test_rbf_model passed]")

    def test_linear_func(self):
        """
            test whether linear function fn is RBF-modeled (cubic, 10/100 observations, with/without polynomial tail)
            the same way in Python and in R. Uses new RNG my_rng2 (avoid cycles!)
        """
        d = 2
        ngrid = 100
        for nobs in [10, 100]:
            for deg in [-1, 1]:
                self.val = 24  # setting for my_rng2
                print(f"\n[test_linear_func] started with d = {d}, deg = {deg}, nobs = {nobs}")
                # rng = np.random.default_rng()
                # xobs = 2*rng.random((nobs,d))-1
                # xobs = my_rng(nobs, d, 24)   # CAUTION: has cycles for n>100, but needed for numerical-equivalence check
                xobs = self.my_rng2(nobs, d)  # better: avoids cycles
                yobs = fn(xobs)
                xgrid = np.mgrid[-1:1:ngrid * 1j, -1:1:ngrid * 1j]
                xflat = xgrid.reshape(d, -1).T
                # print(xflat.shape)  # (ngrid**d, d)
                s = 0
                rbf_model = RBFInterpolator(xobs, yobs, kernel="cubic", degree=deg,
                                            smoothing=0)  # cubic kernel with linear polynomial tail
                yflat = rbf_model(xflat)
                ygrid = yflat.reshape(ngrid, ngrid)

                ytrue = fn(xflat).reshape(ngrid, ngrid)
                delta = ytrue - ygrid
                print(f"avg abs |ytrue - ygrid| = {np.mean(np.abs(delta)):.4e}")

                L = ngrid // 2
                ytest = ygrid[50:60, L - 1]

                np.set_printoptions(7)
                print(ytest)
                # Test numerical equivalence with R-implementation.

                # These are the values generated with demo-rbf3.R and my_rng (DEPRECATED) on the R-side:
                # if nobs == 10:
                #     if deg == -1:
                #         ytest_from_R = np.array([-0.2606985, -0.2238346, -0.1866435, -0.1490981, -0.1111691,
                #                                  -0.0728254, -0.0340340,  0.0052399,  0.0450331,  0.0853837])
                #     elif deg == 1:
                #         ytest_from_R = np.array([-0.0101010,  0.0303030,  0.0707071,  0.1111111,  0.1515152,
                #                                   0.1919192,  0.2323232,  0.2727273,  0.3131313,  0.3535354])
                #     else:
                #         raise NotImplementedError(f"No data from the R side for nobs={nobs} and deg={deg}")
                # elif nobs == 100:
                #     if deg == -1:
                #         ytest_from_R = np.array([-0.0100796,  0.0303247,  0.0707286,  0.1111322,  0.1515357,
                #                                   0.1919392,  0.2323427,  0.2727461,  0.3131495,  0.3535529])
                #     elif deg == 1:
                #         ytest_from_R = np.array([-0.0101010,  0.0303030,  0.0707071,  0.1111111,  0.1515152,
                #                                   0.1919192,  0.2323232,  0.2727273,  0.3131313,  0.3535354])
                #     else:
                #         raise NotImplementedError(f"No data from the R side for nobs={nobs} and deg={deg}")
                # else:
                #     raise NotImplementedError(f"No data from the R side for nobs={nobs}")

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
                print(f"avg |ytest - ytest_from_R| = {np.mean(np.abs(delta)):.4e}")
                assert np.allclose(ytest, ytest_from_R, rtol=1e-4)
                print("[test_linear_func passed]")

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
        x = np.zeros((n,d), dtype=np.float32)
        for n_ in range(n):
            for d_ in range(d):
                self.val = (self.val * self.val * np.sqrt(self.val) + OFS) % MOD  # avoid cycles (!)
                x[n_,d_] = 2*self.val/MOD - 1    # map val to float range [-1,1[
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
    x = np.zeros((n,d), dtype=np.float32)
    for n_ in range(n):
        for d_ in range(d):
            val = (val*val) % MOD           # CAUTION: This might create cycles for larger n (!)
            x[n_,d_] = 2*val/MOD - 1    # map val to float range [-1,1[
    return x




if __name__ == '__main__':
    unittest.main()
