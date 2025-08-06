------------
Optimization
------------

This chapter gives an overview over the optimization process.

Phase I
-------

TODO


Phase II
--------

.. autoclass:: cobraPhaseII.CobraPhaseII
   :members: get_cobra, get_p2, start

.. autoclass:: phase2Vars.Phase2Vars

.. autofunction:: phase2Funcs.adjustMargins


.. _pEffect-label:

The p-Effect
-------------
Functions with both very small and very large slopes (e.g. :math:`f(x)=exp(x^2)`) are difficult to model accurately by
RBF models: The models tend to oscillate. It is much better to squash the function with a log-like transform to a
smaller slope range. On the other hand, for very simple functions like  :math:`f(x)=x` it is disadvantageous to squash
them, because a linear function becomes non-linear and thus harder to model.

The *p-effect* is a number :math:`p_\text{eff}` which allows to decide in each iteration whether to model the objective function :math:`f()`
directly or whether to model :math:`plog(f())`. Here

.. raw:: latex html

   \[	plog(y) =   \begin{cases}
								 +\ln( 1+\bar{y}) & \mbox{ if } \quad \bar{y} \geq 0 \\
								 -\ln( 1-\bar{y}) & \mbox{ if } \quad \bar{y}   <  0
                    \end{cases}  \]

is a strictly monotonic squashing function, where
:math:`\bar{y}=y-p_{shift}` with default ``pshift = 0``.

Given the set :math:`S` of already evaluated points in input space and given :math:`M_f, M_p` as the surrogate models
for :math:`f(), plog(f())` when using all points in :math:`S`,  we calculate
for a new infill point :math:`\vec{x}_{k}` the ratio

.. raw:: latex html

    \[ p_k =      \left\{ \frac{\left\| M_f(\vec{x}_{k})-f(\vec{x}_{k})\right\| }{\left\| plog^{-1}\left(M_p(\vec{x}_{k})\right)-f(\vec{x}_{k})\right\|}
                  \right\}	\]`

If :math:`p_k>1`, then :math:`M_p` has the smaller error. If :math:`p_k<1`, then :math:`M_f` has the smaller error.

For a more stable decision we define the *p-effect* number as

.. raw:: latex html

    \[ p_\text{eff} = \log_{10}(\text{median}\{ p_k \})	\]`

and decide to build :math:`M_p` if :math:`p_\text{eff}>0` and to build :math:`M_f` else.

The calculation of :math:`p_\text{eff}` is done in :meth:`.Surrogator.calcPEffect`.
The conditional application of :math:`plog()` is done in :class:`.Surrogator.AdFitter`.

.. _refineStep-label:

The Refine Step
----------------

The refine step is part of **SACOBRA_Py**'s equality handling mechanism and is executed if the COP contains equality
constraints and ``EQU.active = EQU.refine = True``.

SACOBRA starts with an initially large artificial feasibility band :math:`\mu` around each equality constraint. After a
sequential optimization step, the infill point will be usually at the rim of the feasibility band. In order to refine
the solution, we run a refine optimization with objective

.. raw:: latex html

   \[  \text{Minimize}   \sum_i \left(\max(0,g_i(x)\right)^2 + \sum_j \left(h_j(x)\right)^2  \]`

In the case of one equality constraints, this projects from the rim of the feasibility band to the central equality
surface. In the case of multiple constraints, things get of course more involved, but usually the refine step does a
very good job in reducing the maximum constraint violation. This is important when the feasibility band :math:`\mu` is
reduced for the next iteration: Otherwise, a solution *'on the rim'* that was (artificially) feasible in the current
iteration, might get lost in the next iteration because it is then infeasible.

The refine step is carried out in :meth:`.EvaluatorReal.equ_refine` which is called from :meth:`.EvaluatorReal.update`.

Details Phase II
-----------------

.. autoclass:: randomStarter.RandomStarter
   :members: __init__, random_start

.. _AdFitter-label:

.. autoclass:: surrogator.Surrogator.AdFitter
   :members: __init__, __call__

.. autoclass:: surrogator.Surrogator
   :members: calcPEffect, trainSurrogates

.. autoclass:: rbfModel.RBFmodel
   :members: __init__, __call__

.. autoclass:: rbfSacobra.RBFsacob
   :members: __init__, __call__

.. autoclass:: evaluatorReal.EvaluatorReal
   :members: update, equ_refine, equ_num_max_viol, ine_num_max_viol
