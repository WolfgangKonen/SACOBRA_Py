from typing import Union

import numpy as np

from cobraInit import CobraInitializer
from innerFuncs import verboseprint
from rbfModel import RBFmodel


def concat(a, b):
    return np.concatenate((a, b), axis=None)


class RI2:
    """
    A class to perform the repair of infeasible solutions with method RI2,
    see :meth:`RI2.repairInfeasRI2` for details.

    See :class:`.RIoptions` for all repair-infeasible options and for the definition of ``eps``-**feasibility**.

    :param cobra: an initialized Cobra object
    """
    def __init__(self, cobra: CobraInitializer):
        self.s_opts = cobra.sac_opts
        self.fn = cobra.sac_res['fn']
        self.rw = cobra.rw
        self.rng = cobra.rng
        self.is_equ = cobra.sac_res['is_equ']
        self.equ_ind = np.flatnonzero(self.is_equ)      # index to all equality constraints in constrSurr
        self.equ2Index = concat(self.equ_ind, cobra.sac_res['nConstraints'] + np.arange(self.equ_ind.size))
        # equ2Index: index to all inequalities that stem from equality constraints
        self.currentMu = cobra.sac_res['muVec'][-1]
        self.con_s = None
        self.constrSurr = None

    def repairInfeasRI2(self, x: np.ndarray, gReal: np.ndarray, constrSurr: RBFmodel,
                        cobra: CobraInitializer, currentMu, checkIt, true_grad=None) -> np.ndarray[float]:
        """
        Repair an infeasible solution with method RI2.

        If the solution ``x`` is infeasible, i.e. if there is any ``i`` or any ``j`` such that
        :math:`g_i(x) > 0` or :math:`|h_j(x)| - \\mu > 0`, then:

        1. Estimate the gradient of the constraint surrogate function(s) (go a tiny step in each dimension
           in the direction of constraint increase).
        2. Take ``RI.mmax`` random realizations in the *feasible parallelepiped* and select among them the
           best feasible solution, based on the surrogates.
        3. Check whether the new solution is for every dimension in the bounds ``[lower, upper]`` of the search region.
           If not, set the gradient to 0 in these dimensions and re-iterate from step 2.

        There is no guarantee but a good chance, that the returned solution ``z`` will be feasible.

        For further details see:

        [Koch15a] Koch, P.; Bagheri, S.; Konen, W. et al. "A New Repair Method For Constrained
        Optimization". Proc. 17th Genetic and Evolutionary Computation Conference (GECCO), 2015.

        :param x:       an infeasible solution vector of dimension ``d``
        :param gReal:   a vector :math:`(g_1(x), \ldots, g_m(x), h_1(x), \ldots, h_r(x))`
        :param constrSurr: the constraint surrogate models
        :param cobra:   an object of class :class:`.CobraInitializer`, we need here: ``lower``, ``upper``,
                        :class:`.RIoptions` ``s_opts.RI``
        :param currentMu:  margin for equality constraints
        :param checkIt: if True, perform a check whether the returned solution is really
                      feasible. Needs access to the true constraint functions.
        :param true_grad: (optional) if not None and if checkIt=True, assert that the calculated ``grad_mat`` and
                      ``true_grad`` are close
        :return: ``z``, a vector of dimension ``d`` with a repaired (hopefully feasible) solution
        """
        verboseprint(self.s_opts.verbose, important=False, message="RI2: repairing the infeasible result ...")
        self.currentMu = currentMu      # update
        self.constrSurr = constrSurr    # update
        ri = self.s_opts.RI
        lowerP = cobra.sac_res['lower']
        upperP = cobra.sac_res['upper']
        gradEps = np.min(upperP-lowerP)/1000 if ri.gradEps is None else ri.gradEps
        # gradEps = stepsize for numerical gradient calculation,
        # e.g. 0.002 if the smallest length of search cube is 2.0

        dim = x.size
        nd = np.tile(x,(2*dim+1,1))
        for i in range(1,dim+1):
            nd[2*i-1, i - 1] -= gradEps
            nd[2*i  , i - 1] += gradEps
        # nd is a matrix [[x[0]  , x[1]  , ...],
        #                 [x[0]-e, x[1]  , ...],
        #                 [x[0]+e, x[1]  , ...],
        #                 [x[0]  , x[1]-e, ...],
        #                 [x[0]  , x[1]+e, ...],
        #                 ... ]  with e = gradEps
        ind_n = [2 * (i + 1) - 1 for i in range(0, dim)]    # index to the '-e' entries
        ind_p = [2 * (i + 1)     for i in range(0, dim)]    # index to the '+e' entries
        f = np.apply_along_axis(self._con_s_base,
                                arr=nd, axis=1)       # f.shape = (arr.shape[0],)
        # f is at first a (2*dimension+1 x nconstraint) matrix containing constraint surrogate
        # responses at x (row 0) and at small '-' and '+' deviations from x for any dimension
        # in rows 1, ..., 2 * dim

        if self.s_opts.EQU.active:
            gReal = concat(gReal, -gReal[self.equ_ind])
            f = concat(f, -f[:, self.equ_ind])
            gReal[self.equ2Index] -= self.currentMu
            f[:, self.equ2Index] -= self.currentMu
            # now f is a ((2*dimension+1) x (nconstraint+nequ)) matrix ( nequ = # equality constraints )

        assert f.ndim == 2 and gReal.ndim == 1
        assert f.shape[1] == gReal.size, "Columns do not match in f and gReal"
        assert np.any(gReal+ri.eps1 > 0), "No constraint is eps1-infeasible"

        ix = np.repeat(False, dim)
        # ix: a boolean vector of length dim. It indicates which dimensions of the gradient
        # should be set to zero (because a repair in this dimension would cause the repaired solution
        # to leave the search region [lowerP,upperP]). Initially, all elements of ix are False, i.e.
        # every dimension is taken.
        # [ix is the equivalent to matrix multiplication with E, where E is the identity matrix with 0 at the
        #  diagonal elements indexed by ix==True.]

        while True:
            del_mat = np.zeros((0,dim))
            grad_mat = np.zeros((0,dim))
            viol_lst = []
            for k in range(f.shape[1]):
                if gReal[k] + ri.eps1 > 0:  # if the kth constraint is ri.eps1-infeasible
                    gradf = (f[ind_p, k] - f[ind_n, k]) / (2 * gradEps)
                    gradf[ix] = 0   # zero all dimensions which led to 'out-of-search-region' in previous iterations
                    g2 = np.sum(gradf * gradf)
                    if g2 == 0:
                        msg1 = "Cannot repair infeasible solution w/o moving out of search region"
                        msg2 = " --> will return the incoming (infeasible) solution"
                        verboseprint(self.s_opts.verbose, important=True, message=msg1+msg2)
                        return x
                    del_k = - (gReal[k] + ri.eps1) * gradf / g2
                    del_mat = np.vstack((del_mat, del_k))
                    # del_mat is a matrix with dim columns and as many rows as there are eps1-infeasible constraints.
                    # The kth row of del_mat contains the repair step for the kth constraint.
                    grad_mat = np.vstack((grad_mat, gradf))
                    # Similarly, the kth row of grad_mat contains the gradient for the kth constraint.
                    viol_lst = viol_lst + [k]
                    # viol_lst is the list of violated constraint numbers

            if checkIt:
                self.check_single_constr(x, del_mat, grad_mat, viol_lst, ri.eps1, true_grad)
                print(np.sqrt(np.sum(del_mat*del_mat, axis=1)))  # row sum (along axis 1)

            #  this is the repairInfeasible mechanism after 2014-09-29, see
            #  Notes.d/presentation/present-Wolfgang-2014-09-24-RepairInfeas2:
            num_k = del_mat.shape[0]
            A = self.rng.random((ri.mmax, num_k)) * ri.q
            R = np.matmul(A, del_mat)   # R: matrix with parallelepiped vectors {r_i | row indices i}
            Rx = x + R                  # 'x + R': array broadcasting will repeat x row-wise
            ind_s = np.apply_along_axis(lambda x: self.is_epsilon_feasible(x, ri.eps2, constrSurr),
                                        arr=Rx, axis=1)
            S = R[ind_s, :]             # S: matrix with all eps2-feasible points from R (if any)
            if S.shape[0] == 0:
                # no ri.eps2-feasible point - return the best infeasible solution instead:
                r_best = self.find_best_infeasible(x, R, ri.eps2, constrSurr)
                #if (checkIt) checkBestInfeasible(Del);
            else:
                # select the best ri.eps2-feasible point from S
                r_best = self.select_best(S)

            z = x + r_best
            ix2 = np.flatnonzero((z > upperP) | (z < lowerP))
            if ix2.size == 0:
                # we are done: z is inside search region in every dimension
                break       # out of while
            ix = ix2 | ix        # don't forget the dimensions which were 'outside' in previous iterations

        if checkIt: self.check_solution(z, gReal)
        return z

    # ----------------------------------------------------------------------------------- #
    # --- private methods and helper methods for repairInfeasRI2               ---------- #
    # ----------------------------------------------------------------------------------- #
    def _con_s_base(self, x):
        if self.s_opts.RI.trueFuncForSurrogates:
            con_s = self.fn(x)[1:]  # true constraints
        else:
            con_s = self.constrSurr(x)[0, :]  # constraint surrogates
        return con_s

    def _calc_con_s(self, x):
        con_s = self._con_s_base(x)
        if self.s_opts.EQU.active:
            con_s[self.equ_ind] = np.abs(con_s[self.equ_ind]) - self.currentMu
        return con_s

    def is_epsilon_feasible(self, x: np.ndarray, eps2: float, constrSurr: RBFmodel):
        """
        Return True, if all constraint surrogates have a feasibility 'better' than ``eps2`` (i.e. they are
        ``-eps2`` or lower)

        :param x:           point in input space, vector of dimension ``d``. If ``ID.rescale==True`` then ``x``
            should be (forward-) rescaled.
        :param eps2:        safety margin
        :param constrSurr:  constraint surrogate models
        :return: whether ``x`` is ``eps2``-feasible
        """
        self.constrSurr = constrSurr    # update
        self.con_s = self._calc_con_s(x)
        return np.all(self.con_s <= -eps2)

    def find_best_infeasible(self, x, R, eps2: float, constrSurr: RBFmodel):
        """
        Given a set of ``k=0,...,K-1`` points ``z[k] = x+R[k,:]``, find this point ``z[kBest]`` which has

        1. the lowest number of constraints being **not** ``eps2``-feasible, and,
        2. if there is more than one such point, select among them the point with the lowest maximum violation.

        :param x:          point in input space, vector of dimension ``d``. If ``ID.rescale==True`` then ``x``
            should be (forward-) rescaled.
        :param R:          ``(K,d)``-matrix where row ``k`` holds delta vector ``z[k]-x``
        :param eps2:       safety margin
        :param constrSurr: constraint surrogate models
        :return: the best residual ``R[kBest,:]`` = ``z[kBest]-x``
        """
        self.constrSurr = constrSurr    # update
        def numMaxViol(z):
            con_s = self._calc_con_s(z)
            ind = np.flatnonzero(con_s + eps2 > 0)
            # print(ind.size, max(con_s), con_s + eps2)
            return ind.size , max(con_s)

        result = np.apply_along_axis(numMaxViol, arr=R+x, axis=1)       # result.shape = (arr.shape[0],2)
        numViol = np.int32(result[:,0])
        maxViol = result[:,1]
        # DBG = True
        # if DBG:
        #     x0 = x + R[0,:]
        #     isf = self.is_epsilon_feasible(x0, eps2, constrSurr)
        #     print(self.con_s)
        maxViol[numViol > min(numViol)] = np.inf    # invalidate all entries with too high numViol
        kBest = np.flatnonzero(maxViol == min(maxViol))[0]
        return R[kBest, :]

    def select_best(self, S: np.ndarray):
        """
        Among the ``eps2``-feasible points (rows of S) select the one with minimal length

        :param S: matrix with all ``eps2``-feasible delta vectors in its rows
        :return: point with minimal length
        """
        l2 = np.sum(S*S, axis=1)    # row sum (along axis 1)
        ind = np.flatnonzero(l2 == min(l2))[0]  # if several points have the same minimal length, select the first one
        return S[ind, :]

    def check_single_constr(self, x, del_mat, grad_mat, viol_lst, eps, true_grad: Union[np.ndarray, None]):
        """
        Debug only: Does the single repair step ``del_mat[k,:]`` calculated for violated constraint ``k`` with
        surrogate ``s_k`` bring the infill point ``x`` close to the ``eps``-boundary of constraint ``k``, as it should?

        If so, ``fBefore[k]+eps`` should be much larger than ``fSingle[k]+eps`` in magnitude.

        With ``fBefore[k]=s_k(x)`` and ``fSingle[k]=s_k(x+del_mat[k,:])``.

        In addition, we assert that ``grad_mat`` is close to ``true_grad`` (if ``true_grad`` is not None).

        :param x:
        :param del_mat:
        :param grad_mat:
        :param viol_lst:
        :param eps:
        :param true_grad:
        """
        if true_grad is not None:
            assert np.allclose(grad_mat, true_grad), "grad_mat is not close to true_grad"
        print(f"Length of repair steps: {np.sqrt(np.sum(del_mat*del_mat, axis=1))}")    # row sum (along axis 1)
        print(f"Length of gradients: {np.sqrt(np.sum(grad_mat * grad_mat, axis=1))}")   # row sum (along axis 1)
        fBefore = np.zeros(len(viol_lst))
        fSingle = np.zeros(len(viol_lst))
        for k, v in enumerate(viol_lst):
            fBefore[k] = self._calc_con_s(x)[v]
            # fBefore[k] is the constraint value of constraint k at x (prior to repair step)
            fSingle[k] = self._calc_con_s(x + del_mat[k,:])[v]
            # fSingle[k] is the constraint value of constraint k at 'x + repair step'. Should be close to -eps
        np.set_printoptions(precision=8)
        print(f"Violations + eps before single corrections: {fBefore + eps}")
        print(f"Violations + eps after  single corrections: {fSingle + eps}")
        print(f"Quotient {(fBefore + eps)/(fSingle + eps)} (should be larger than approx. 8.0 in magnitude)")
        np.set_printoptions(precision=8)
        assert np.allclose(fSingle, -eps)

    def check_solution(self, z, gReal):
        ri = self.s_opts.RI
        fTrue = self.fn(z)[1:]  # fTrue: true constraint values after repair
        fRbf = self.constrSurr(z)[0, :]  # fRbf:  constraint surrogate values after repair
        # print(gReal); print(fRbf); print(fTrue)
        violatedConstraints = np.flatnonzero(fTrue > 0)
        cfcReal = maxReal = 0
        if np.any(fTrue > 0):
            cfcReal = np.sum(fTrue[violatedConstraints])
            maxReal = np.max(fTrue[violatedConstraints])
        print(f"Repaired solution is feasible:  {cfcReal <= 0}, cfcReal={cfcReal}, maxViol={maxReal}")
        print(f"Repaired solution is eps1-feasible:  {np.all(fTrue + ri.eps1 <= 0)}, fTrue+eps1={fTrue + ri.eps1}")
        print(f"   eps1-inf constraints before repair: {np.flatnonzero(gReal + ri.eps1 > 0)}")
        print(f"   violated constraints before repair: {np.flatnonzero(gReal > 0)}")
        print(f"   violated constraints  after repair: {np.flatnonzero(fTrue>0)}")
        print(f"   violated c-surrogates after repair: {np.flatnonzero(fRbf>0)}")
