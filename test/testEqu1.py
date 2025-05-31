import unittest
import numpy as np
from cobraInit import CobraInitializer
from cobraPhaseII import CobraPhaseII
from opt.equOptions import EQUoptions
from opt.rbfOptions import RBFoptions
from opt.seqOptions import SEQoptions
from rescaleWrapper import RescaleWrapper
from opt.idOptions import IDoptions
from opt.sacOptions import SACoptions
from opt.trOptions import TRoptions

verb = 1


class TestEqu1(unittest.TestCase):
    """
        Several tests for equality constraints in SACOBRA
    """
    def test_equ_mu_init(self):
        """  Test that the different options to initialize :math:`\\mu =` ``currentMu`` work as expected
        """
        nobs = 10
        x0 = np.array([2.5, 2.4])
        lower = np.array([-5, -5])
        upper = np.array([5, 5])
        idp = 2*x0.size + 1
        initTVec = ["useGrange", "TAV", "TMV", "EMV"]   # the allowed options for cobra.sac_opts.EQU.initType

        # the following dict muInitDict_from_R is copied from the output of demo-equMuInit.R:
        muInitDict_from_R = {1: [13.57579269,   3.44648875,   2.22324437,   2.22324437],
                             10: [135.75792695,  43.46488746,  22.23244373,  22.23244373]}

        for hnfac in [1,10]:
            # simple problem with two equality constraints
            # (one equ only would result in "TAV", "TMV" and "EMV" having all the same value):
            def fn(x):
                return np.array([3 * np.sum(x ** 2), np.sum(x * hnfac + 1) - 2, np.sum(x * hnfac) - 1])

            muInitVec = np.zeros(len(initTVec))
            for i, initT in enumerate(initTVec):
                cobra = CobraInitializer(x0, fn, "fName", lower, upper,
                                         is_equ=np.array([True, True]),
                                         s_opts=SACoptions(verbose=verb, feval=2*nobs,
                                                           ID=IDoptions(initDesign="RAND_R", initDesPoints=idp),
                                                           EQU=EQUoptions(initType=initT)))
                sac_res =  cobra.get_sac_res()
                muInit = sac_res['muVec'][0]

                #print("test_equ_mu_init:", fn(x0))
                print(f"mu_init[{initT}] = {muInit}")
                muInitVec[i] = muInit
            print(muInitVec)
            muInitVec_from_R = muInitDict_from_R[hnfac]
            assert np.allclose(muInitVec, muInitVec_from_R)
        print("[test_equ_mu_init passed]")

    def test_currentEps(self):
        """  Test the values generated for vector currentMu = muVec.
             Compare with the exact values generated on the R side.
        """
        nobs = 10
        x0 = np.array([2.5, 2.4])
        lower = np.array([-5, -5])
        upper = np.array([5, 5])
        idp = 6

        muTypes = ["expFunc", "SAexpFunc", "funcDim", "funcSDim", "Zhang", "CONS"]
        # the allowed options for cobra.sac_opts.EQU.muType

        # Matrix muMat_from_R is copied from muMat of method demo_currentEps in demo-equMuInit.R.
        # It has one row for each value of muTypes (and 5 columns, since we will compare below the first five iterations
        # ipd:idp+5 of phase II):
        muMat_from_R = np.array([
            [3.929555,  2.6197033, 1.7464689, 1.1643126, 0.7762084],
            [3.929555,  1.5769679, 0.7756560, 0.5085520, 0.4195173],
            [3.929555,  2.6197034, 1.7464690, 1.1643127, 0.7762085],
            [3.929555,  3.2084683, 2.6197034, 2.1389789, 1.7464690],
            [3.929555,  0.0000001, 0.0000001, 0.0000001, 0.0000001],
            [0.0000001, 0.0000001, 0.0000001, 0.0000001, 0.0000001]
        ])
        # [Note that all rows but the last start with the same value 3.929555, because we start always with
        # EQU.initType='TAV', only EQU.muType=muT changes. The last row (muT="CONS") has a different initial value,
        # because in this case the initial value is overwritten with EQU.muFinal.]

        # a simple problem with two equality constraints:
        def fn(x):
            return np.array([3 * np.sum(x ** 2), np.sum(x + 1) - 2, np.sum(x) - 1])

        muMat = None
        for i, muT in enumerate(muTypes):
            cobra = CobraInitializer(x0, fn, "fName", lower, upper, np.array([True, True]),
                                     s_opts=SACoptions(verbose=verb, feval=15,
                                                       ID=IDoptions(initDesign="RAND_R", initDesPoints=idp),
                                                       EQU=EQUoptions(muType=muT)))
            sac_res = cobra.get_sac_res()
            sac_opts = cobra.get_sac_opts()

            c2 = CobraPhaseII(cobra)
            p2 = c2.get_p2()
            c2.start()

            cobra = c2.get_cobra()
            s_res = cobra.get_sac_res()
            muVec = s_res['muVec'][idp:idp+5]
            muMat = muVec if muMat is None else np.vstack((muMat,muVec))
            print(muT, muVec)
            assert np.allclose(muVec, muMat_from_R[i,:], atol=1e-4), f"currentMu assertion fail for type muT = {muT}"

        np.set_printoptions(precision=5)
        print(muMat)
        np.set_printoptions(precision=8)
        print("[test_equ_mu_init passed]")

    def test_refine1(self):
        """  Test the refine step (1 equality constraint)
        """
        x0 = np.array([2.5, 2.4])
        lower = np.array([-5, -5])
        upper = np.array([5, 5])
        d = lower.size
        idp = 6

        for hnfac in [1]:  # [1, 10, 100]:
            def fn(x):
                return np.array([3 * np.sum(x ** 2), np.sum(x * hnfac) - 2])

            cobra = CobraInitializer(x0, fn, "fName", lower, upper, np.array([True]),
                                     s_opts=SACoptions(verbose=verb, feval=30,
                                                       ID=IDoptions(initDesign="RAND_R", initDesPoints=idp),
                                                       RBF=RBFoptions(degree=1),
                                                       EQU=EQUoptions(refine=True, refinePrint=True),
                                                       SEQ=SEQoptions(finalEpsXiZero=True)))
            c2 = CobraPhaseII(cobra)
            p2 = c2.get_p2()
            c2.start()

            if hnfac == 1:
                xbest = cobra.get_xbest()   # [1.00002489 0.99997512]
                xbest_from_R = np.array([0.9994554, 1.0005446])
                fbest = cobra.get_fbest()   # 6.000000080139946
                fbest_from_R = 6.00001011
            elif hnfac == 10:
                xbest = cobra.get_xbest()   # [0.09991922 0.10008079]
                xbest_from_R = np.array([0.0997111, 0.1002889])
                fbest = cobra.get_fbest()   # 0.06000004269797094
                fbest_from_R = 0.0600005008
            print(f"test_refine1 xbest: {xbest}")
            print(f"test_refine1 fbest: {fbest}")
            assert np.allclose(xbest, xbest_from_R, atol=1e-3)
            assert np.allclose(fbest, fbest_from_R)
        print("[test_refine1 passed]")

    def test_refine2(self):
        """  Test the refine step (2 equality constraints)
        """
        x0 = np.array([2.5, 2.4])
        lower = np.array([-5, -5])
        upper = np.array([5, 5])
        d = lower.size
        idp = 6

        for hnfac in [1,10]:  # [1, 10, 100]:
            def fn(x):
                return np.array([3 * np.sum(x ** 2), x[0] - x[1] - 1, np.sum(x * hnfac) - 2])

            feval = 50 #   30 if hnfac==1 else 40
            cobra = CobraInitializer(x0, fn, "fName", lower, upper,
                                     is_equ=np.array([True, True]),
                                     s_opts=SACoptions(verbose=verb, feval=feval,
                                                       RBF=RBFoptions(degree=1),
                                                       ID=IDoptions(initDesign="RAND_R", initDesPoints=idp),
                                                       EQU=EQUoptions(refine=False, refinePrint=True),
                                                       SEQ=SEQoptions(finalEpsXiZero=True)))
            c2 = CobraPhaseII(cobra)
            p2 = c2.get_p2()
            c2.start()

            if hnfac == 1:
                xbest = cobra.get_xbest()   # [1.50000015 0.5]
                xbest_from_R = np.array([1.5, 0.5])
                fbest = cobra.get_fbest()   # 7.500001393073267
                fbest_from_R = 7.5
            elif hnfac == 10:
                xbest = cobra.get_xbest()   # [ 0.60000003 -0.39999987]
                xbest_from_R = np.array([0.5999986, -0.3999986])
                fbest = cobra.get_fbest()   # 1.5599997818834788
                fbest_from_R = 1.559992
            print(f"test_refine2 xbest: {xbest}")
            print(f"test_refine2 fbest: {fbest}")
            assert np.allclose(xbest, xbest_from_R)
            assert np.allclose(fbest, fbest_from_R)
        print("[test_refine2 passed]")


if __name__ == '__main__':
    unittest.main()
