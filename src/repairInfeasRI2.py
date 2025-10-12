import numpy as np

from cobraInit import CobraInitializer
from innerFuncs import verboseprint
from rbfModel import RBFmodel


def concat(a, b):
    return np.concatenate((a, b), axis=None)


class RepairInfeasibleRI2:
    def __init__(self, cobra: CobraInitializer, currentMu):
        self.s_opts = cobra.sac_opts
        self.fn = cobra.sac_res['fn']
        self.rw = cobra.rw
        self.is_equ = cobra.sac_res['is_equ']
        self.equ_ind = np.flatnonzero(self.is_equ)      # index to all equality constraints in constrSurr
        self.equ2Index = concat(self.equ_ind, cobra.sac_res['nConstraints'] + np.arange(self.equ_ind.size))
        self.currentMu = currentMu
        self.con_s = None

    def repairInfeasRI2(self, x: np.ndarray, gReal: np.ndarray, constrSurr: RBFmodel,
                        cobra: CobraInitializer, currentMu, checkIt):
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
        :return: ``z``, a vector of dimension ``d`` with a repaired (hopefully feasible) solution
        """
        verboseprint(self.s_opts.verbose, important=False, message="RI2: repairing the infeasible result ...")
        self.currentMu = currentMu      # update
        ri = self.s_opts.RI
        lowerP = cobra.sac_res['lower']
        upperP = cobra.sac_res['upper']
        gradEps = np.min(upperP-lowerP)/1000 if ri.gradEps is None else ri.gradEps
        # gradEps = stepsize for numerical gradient calculation,
        # e.g. 0.001 if the smallest length of search cube is 1.0

        dim = x.size
        nd = np.tile(x,(2*dim+1,1))
        for i in range(1,dim+1):
            nd[2*i-1, i] -= gradEps
            nd[2*i  , i] += gradEps
        # nd is a matrix [[x[0]  , x[1]  , ...],
        #                 [x[0]-e, x[1]  , ...],
        #                 [x[0]+e, x[1]  , ...],
        #                 [x[0]  , x[1]-e, ...],
        #                 [x[0]  , x[1]+e, ...],
        #                 ... ]  with e = gradEps
        f = np.apply_along_axis(self._con_s_base, arr=nd, axis=1)       # f.shape = (arr.shape[0],)
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


    #
    #   while(1) {
    #     Del <- NULL
    #     Grad <- NULL
    #     Viol <- NULL
    #     for (k in 1:ncol(f)) {
    #       if (gReal[k]+ri$eps1>0) {   # if the kth constraint is not ri$eps1-feasible
    #         gradf <- rep(0,dimension)
    #         for (i in 1:dimension) {
    #           if (f[2*i,k]>f[2*i+1,k])    # if the penalty increase is larger in direction '-gradEps':
    #             gradf[i] <- (f[2*i  ,k]-f[1,k])/(-gradEps)
    #           else                        # else, i.e. '+gradEps':
    #             gradf[i] <- (f[2*i+1,k]-f[1,k])/gradEps
    #         }
    #         gradf[ix] <- 0            # zero all dimensions which led to out-of-search-region
    #                                   # repairs in previous iterations
    #         g2 <- sum(gradf*gradf)
    #         if (g2==0) {
    #           warning("Cannot repair infeasible solution w/o moving out of search region")
    #           # Return the incoming (infeasible) solution instead:
    #           z = x
    #           return(z)
    #         }
    #         Del_k =  -gradf * (gReal[k]+ri$eps1)/g2
    #         Del = rbind(Del,Del_k)
    #         # Del is a matrix with as many rows as there are eps1-infeasible constraints and
    #         # with d (input space dimension) columns.
    #         # The kth row of Del contains the step suggested for the kth constraint.
    #         Grad = rbind(Grad,gradf)
    #         # Similarly, the kth row of Grad contains the gradient for the kth constraint.
    #
    #         Viol = c(Viol,k)  # Viol is the list of violated constraint numbers
    #       } # if (gReal[k])
    #     } # for k
    #     #checkSingleConstraints(Del,Grad,Viol);
    #     #browser()
    #     #if (checkIt) print(sqrt(rowSums(Del*Del)));
    #
    #       # this is the new repairInfeasible mechanism (after 2014-09-29, see
    #       # Notes.d/presentation/present-Wolfgang-2014-09-24-RepairInfeas2):
    #       S = NULL
    #       deltaMat = NULL
    #       # just for safety, this should normally not happen:
    #       testit::assert("Del is NULL!",!is.null(Del))
    #       for (m in 1:ri$mmax) {
    #         alpha = runif(nrow(Del))*ri$q         # random coef. from distribution U[0,a]
    #         alphaMat = outer(alpha,rep(1,ncol(Del)))
    #         delta = colSums(alphaMat*Del)
    #         deltaMat = rbind(deltaMat,delta)
    #         if (isEpsilonFeasible(x+delta,ri$eps2,rbf.model))
    #           S = rbind(S,delta)
    #       }
    #       if (is.null(S)) {
    #         # No ri$eps2-feasible point - return the best infeasible solution instead:
    #         Delta = findBestInfeasible(deltaMat,x,ri$eps2,rbf.model)
    #         if (checkIt) checkBestInfeasible(Del);
    #       } else {
    #         Delta = selectBest(S,x,ri)
    #       } # else (is.null(S))
    #       z = x + Delta       # Delta is a vector of dimension d
    #
    #     # This should normally not happen:
    #     testit::assert("New solution z contains NA or NaN!", !any(is.na(z)))
    #
    #     ix2 = (z > upperP) | (z < lowerP)
    #     if (! any(ix2)) {
    #       # we are done: z is inside search region in every dimension
    #       break # out of while
    #     }
    #     ix = ix2 | ix   # don't forget the dimensions which were 'outside' in previous iterations
    #   } # while

    def _con_s_base(self, x, constrSurr):
        if self.s_opts.RI.trueFuncForSurrogates:
            con_s = self.fn(x)[1:]  # true constraints
        else:
            con_s = constrSurr(x)[0, :]  # constraint surrogates
        return con_s

    def _calc_con_s(self, x, constrSurr):
        con_s = self._con_s_base(x, constrSurr)
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
        self.con_s = self._calc_con_s(x, constrSurr)
        return np.all(self.con_s <= -eps2)

    def find_best_feasible(self, x, deltaMat, eps2: float, constrSurr: RBFmodel):
        """
        Given a set of ``k=0,...,K-1`` points ``z[k] = x+deltaMat[k,:]``, find this point ``z[kBest]`` which has

        1. the lowest number of constraints being **not** eps2-feasible, and,
        2. if there is more than one such point, select among them the point with the lowest maximum violation.

        :param x:           point in input space, vector of dimension ``d``. If ``ID.rescale==True`` then ``x``
            should be (forward-) rescaled.
        :param deltaMat:    ``(K,d)``-matrix where row ``k`` holds delta vector ``z[k]-x``
        :param eps2:        safety margin
        :param constrSurr:  constraint surrogate models
        :return: ``delta[kBest,:]``
        """
        def numMaxViol(z):
            con_s = self._calc_con_s(z, constrSurr)
            ind = np.flatnonzero(con_s + eps2 > 0)
            # print(ind.size, max(con_s), con_s + eps2)
            return ind.size , max(con_s)

        result = np.apply_along_axis(numMaxViol, arr=deltaMat+x, axis=1)       # result.shape = (arr.shape[0],2)
        numViol = np.int32(result[:,0])
        maxViol = result[:,1]
        # DBG = True
        # if DBG:
        #     x0 = x + deltaMat[0,:]
        #     isf = self.is_epsilon_feasible(x0, eps2, constrSurr)
        #     print(self.con_s)
        maxViol[numViol > min(numViol)] = np.inf    # invalidate all entries with too high numViol
        kBest = np.flatnonzero(maxViol == min(maxViol))[0]
        return deltaMat[kBest, :]

