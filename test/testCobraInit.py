import unittest
import numpy as np
from cobraInit import CobraInitializer
from cobraPhaseII import CobraPhaseII
from rescaleWrapper import RescaleWrapper
from opt.idOptions import IDoptions
from opt.sacOptions import SACoptions
from opt.trOptions import TRoptions

verb = 1


class TestCobraInit(unittest.TestCase):
    """
        Several tests for component :class:`CobraInitializer`
    """
    def test_fn_rescale(self):
        """  Test rescaling for a specific 1D-function
        """
        def fn(x):
            return np.array([3*np.sum(x ** 2)])
        x0 = np.array([2.5,2.5])
        lower =np.array([-5,-5])
        upper =np.array([ 5, 5])
        cobra = CobraInitializer(x0, fn, "fName", lower, upper, s_opts=SACoptions(verbose=verb))
        sac_res =  cobra.get_sac_res()
        sac_opts = cobra.get_sac_opts()
        rescaler = RescaleWrapper(fn, lower, upper, sac_res['lower'], sac_res['upper'])
        # these assertions should be valid for all values of fn, x0, lower, upper:
        self.assertEqual(sac_res['lower'][0], -1)
        self.assertEqual(sac_res['upper'][0], 1)
        self.assertEqual(sac_res['fn'](sac_res['x0']), fn(x0))
        self.assertTrue((rescaler.inverse(rescaler.forward(x0))==x0).all())
        # these assertions are valid only for the specific values above of fn, x0, lower, upper:
        self.assertEqual(sac_res['fn'](sac_res['x0']), 37.5)
        self.assertTrue((rescaler.forward(x0)==np.array([0.5,0.5])).all())
        print("test_rescale:\n", fn(x0))

    def test_fn_rescale2(self):
        """  Test rescaling for a specific 2D-function
        """
        def fn(x):
            return np.array([3*np.sum(x ** 2), np.sum(x)-1])
        x0 = np.array([2.5,2.5])
        lower =np.array([-5,-5])
        upper =np.array([ 5, 5])
        cobra = CobraInitializer(x0, fn, "fName", lower, upper, s_opts=SACoptions(verbose=verb))
        sac_res =  cobra.get_sac_res()
        sac_opts = cobra.get_sac_opts()
        rescaler = RescaleWrapper(fn, lower, upper, sac_res['lower'], sac_res['upper'])
        # these assertions should be valid for all values of fn, x0, lower, upper:
        self.assertEqual(sac_res['lower'][0], -1)
        self.assertEqual(sac_res['upper'][0], 1)
        self.assertTrue((sac_res['fn'](sac_res['x0'])==fn(x0)).all())
        self.assertTrue((rescaler.inverse(rescaler.forward(x0))==x0).all())
        # these assertions are valid only for the specific values above of fn, x0, lower, upper:
        self.assertTrue((sac_res['fn'](sac_res['x0'])==np.array([37.5,4])).all())
        self.assertTrue((rescaler.forward(x0)==np.array([0.5,0.5])).all())
        print("test_rescale2:\n", fn(x0))
        # arr = np.vstack((x0,x0-2, x0-5))
        # z = np.apply_along_axis(fn,axis=1,arr=arr)
        # print(z.shape)
        # print(z)

    def test_init_design(self):
        """ Test whether InitDesigner produces arrays Fres, Gres that are numerically equivalent
            to what we compute from (rescaled) A and fn
        """
        def fn(x):
            return np.array([3*np.sum(x ** 2), np.sum(x)-1])
        x0 = np.array([2.5,2.4])
        lower =np.array([-5,-5])
        upper =np.array([ 5, 5])
        cobra = CobraInitializer(x0, fn, "fName", lower, upper,
                                 s_opts=SACoptions(verbose=verb, TR=TRoptions(radiInit=0.42)))
        sac_res =  cobra.get_sac_res()
        A = sac_res['A']
        newXStart = sac_res['x0']
        Fres = sac_res['Fres']
        Gres = sac_res['Gres']
        newfn = sac_res['fn']
        # these assertions should be valid for all values of fn, x0, lower, upper:
        for i in range(A.shape[0]):
            fnEval = newfn(A[i,:])
            self.assertEqual(Fres[i], fnEval[0])
            for j in range(Gres.shape[1]):
                self.assertEqual(Gres[i,j], fnEval[1+j])
        fnEval = newfn(newXStart)
        self.assertEqual(Fres[-1], fnEval[0])
        for j in range(Gres.shape[1]):
            self.assertEqual(Gres[-1, j], fnEval[1 + j])
        # self.assertEqual(sac_res['upper'][0], 1)
        # self.assertTrue((sac_res['fn'](sac_res['x0'])==fn(x0)).all())
        print("test_init_design:\n", cobra.get_sac_opts().TR.radiInit)
        print(A)
        print(newXStart)

    def test_init_design_R(self):
        """ same as test_init_design, but with reproducible random numbers to compare with results from R
            (see demo-id.R)
        """
        def fn(x):
            return np.array([3 * np.sum(x ** 2), np.sum(x) - 1])

        x0 = np.array([2.5, 2.4])
        lower = np.array([-5, -5])
        upper = np.array([5, 5])
        idp = 2*x0.size + 1
        cobra = CobraInitializer(x0, fn, "fName", lower, upper,
                                 s_opts=SACoptions(verbose=verb,
                                                   ID=IDoptions(initDesign="RAND_R", initDesPoints=idp)))
        sac_res = cobra.get_sac_res()
        A = sac_res['A']
        newXStart = sac_res['x0']
        Fres = sac_res['Fres']
        Gres = sac_res['Gres']
        # self.assertEqual(sac_res['upper'][0], 1)
        # self.assertTrue((sac_res['fn'](sac_res['x0'])==fn(x0)).all())

        # these are the results computed on the R side (file demo-id.R):
        A_from_R = np.array( [  [ -0.96472247, -0.7704361],
                                [  0.16435849,  0.2802904],
                                [ -0.09018369,  0.1833372],
                                [  0.24347296, -0.3384863],
                                [  0.50000000,  0.4800000]])
        F_from_R = np.array( [114.31958851,   7.91823088,   3.13092105,  13.03890450,  36.03000000])
        G_from_R = np.array([   [-9.67579269],
                                [ 1.22324437],
                                [-0.53423260],
                                [-1.47506675],
                                [ 3.90000000]])
        self.assertTrue(np.allclose(A, A_from_R),"A and A_from_R are not close")
        self.assertTrue(np.allclose(Fres, F_from_R),"Fres and F_from_R are not close")
        self.assertTrue(np.allclose(Gres, G_from_R),"Gres and G_from_R are not close")
        print("test_init_design2:\n", A)
        print(Fres)
        # print(Fres - F_from_R)
        # print(Gres - G_from_R)

        # Result: The numbers do all fulfil the np.allclose assertions.
        # The errors are in most cases < 5*10e-7. In one case (Fres[2]), the error is 1e-6.

    def test_adCon_R(self):
        """ Define a problem fn with two constraints that trigger adCon normalization. Test whether adjustment
            of constraints works as expected, i.e. whether results are numerically equivalent to R (see demo-adCon.R)
        """
        def fn(x):
            return np.array([3 * np.sum(x ** 2), np.sum(x) - 1,  -3000*(np.sum(x)-10)])

        x0 = np.array([2.5, 2.4])
        lower = np.array([-5, -5])
        upper = np.array([5, 5])
        idp = 2*x0.size + 1
        cobra = CobraInitializer(x0, fn, "fName", lower, upper,
                                 s_opts=SACoptions(verbose=verb,
                                                   ID=IDoptions(initDesign="RAND_R", initDesPoints=idp)))
        sac_res = cobra.get_sac_res()
        A = sac_res['A']
        newXStart = sac_res['x0']
        Fres = sac_res['Fres']
        Gres = sac_res['Gres']
        # self.assertEqual(sac_res['upper'][0], 1)
        # self.assertTrue((sac_res['fn'](sac_res['x0'])==fn(x0)).all())

        # these are the results computed on the R side (file demo-id.R):
        A_from_R = np.array( [  [ -0.96472247, -0.7704361],
                                [  0.16435849,  0.2802904],
                                [ -0.09018369,  0.1833372],
                                [  0.24347296, -0.3384863],
                                [  0.50000000,  0.4800000]])
        F_from_R = np.array( [114.31958851,   7.91823088,   3.13092105,  13.03890450,  36.03000000])
        # this is the new Gres after constraint normalization (both columns have the same max-min-range):
        G_from_R = np.array([   [-14518.52693811, 28023.02693811],
                                [ 1835.47818153, 11669.02181847],
                                [-801.61602188, 14306.11602188],
                                [-2213.33765136, 15717.83765136],
                                [ 5851.95000000, 7652.55000000]])

        self.assertTrue(np.allclose(A, A_from_R),"A and A_from_R are not close")
        self.assertTrue(np.allclose(Fres, F_from_R),"Fres and F_from_R are not close")
        self.assertTrue(np.allclose(Gres, G_from_R),"Gres and G_from_R are not close")
        # test that all columns of new Gres have the same max-min-range:
        GRL = np.apply_along_axis(self.maxMinLen, axis=0, arr=Gres)
        self.assertTrue(np.allclose(GRL[0], GRL[1]), "GRL is not the same for the (normalized) constraints")
        for i in range(A.shape[0]):
            # test that sac_res['fn'] is appropriately scaled such that each row of A produces the corresponding
            # row of the (new, normalized) Gres:
            x = A[i,:]
            y = np.hstack((Fres[i], Gres[i,:]))
            # print((y - sac_res['fn'](x))/y)
            self.assertTrue(np.allclose(sac_res['fn'](x), y))
        print("test_adCon:\n", Gres)
        # print(Fres)
        # print(Fres - F_from_R)
        # print((Gres - G_from_R)/Gres)

        # Result: The numbers in Gres (after constraint normalization) do all fulfill the np.allclose assertion.
        # The *relative* errors are in all cases < 5*10e-7.

    def test_phaseII(self):
        def fn(x):
            return np.array([3 * np.sum(x ** 2), np.sum(x) - 1,  -3000*(np.sum(x)-10)])
        x0 = np.array([2.5, 2.4])
        lower = np.array([-5, -5])
        upper = np.array([5, 5])
        cobra = CobraInitializer(x0, fn, "fName", lower, upper,
                                 s_opts=SACoptions(verbose=verb, ID=IDoptions(initDesign="RAND_R")))
        print("\ntest_phaseII:")
        assert cobra.phase == "init"
        print(cobra.sac_opts.ISA.TGR)
        c2 = CobraPhaseII(cobra)
        cobra = c2.get_cobra()
        assert cobra.phase == "phase2"
        print(cobra.sac_opts.ISA.TGR)

    def maxMinLen(self, x):
        maxL = max(x)
        minL = min(x)
        return maxL - minL


if __name__ == '__main__':
    unittest.main()
