------------
Optimization
------------

This chapter gives an overview over the optimization process. SACOBRA's optimization process consists of two phases
Phase I and Phase II, where Phase I is optional.

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
for a new infill point :math:`\vec{x}_{new}` the ratio

.. raw:: latex html

    \[ p_k =      \left\{ \frac{\left\| M_f(\vec{x}_{new})-f(\vec{x}_{new})\right\| }{\left\| plog^{-1}\left(M_p(\vec{x}_{new})\right)-f(\vec{x}_{new})\right\|}
                  \right\}	\]`

If :math:`p_k>1`, then :math:`M_p` has the smaller error. If :math:`p_k<1`, then :math:`M_f` has the smaller error.

For a more stable decision we define the *p-effect* number as

.. raw:: latex html

    \[ p_\text{eff} = \log_{10}(\text{median}\{ p_k \})	\]`

and decide to build :math:`M_p` if :math:`p_\text{eff}>0` and to build :math:`M_f` else.

The calculation of :math:`p_\text{eff}` is done in :meth:`.Surrogator1.calcPEffect`.
The conditional application of :math:`plog()` is done in :class:`.Surrogator1.AdFitter`.

.. _detail_onlinePLOG-label:

Details for onlinePLOG
______________________

The above p-effect recipe is realized in case ``ISA.onlinePLOG =`` **O_LOGIC.XNEW**. It tracks online (in each
iteration) which of the two objective surrogate models is better. It uses the new infill point :math:`\vec{x}_{new}`
(found after sequential optimization), which is usually different from all earlier points.
But a closer look reveals that there are two drawbacks:

- The p-effect can only be calculated **after** the sequential optimization of the current iteration and so it can be
  only used for the model decision in the **next** iteration, which might be confusing.
- It is not guaranteed that the new infill point :math:`\vec{x}_{new}` is different from all earlier points. Especially
  if the distance requirement is 0, :math:`\vec{x}_{new}` may be very close or identical to earlier points. Then both
  approximation errors will be very small or even 0, which causes the error ratio to be unreliable.

An alternative (better) p-effect recipe is the case ``ISA.onlinePLOG =`` **O_LOGIC.MIDPTS**. In this case we replace
:math:`\vec{x}_{new}` with a set of points, namely all :math:`P=p(p-1)/2` midpoints :math:`\vec{m}_{\ell}` between the :math:`p =`
``ISA.pEff_npts`` first points of initial design matrix ``A``. This has the advantage that the p-effect can be calulated
directly after training the surrogate models :math:`M_f, M_p` (because the midpoints are known in advance) and that the
new decision number

.. raw:: latex html

    \[ p_k =      \left\{ \frac{\sum_{\ell=1}^P{\left\| M_f(\vec{m}_{\ell})-f(\vec{m}_{\ell})\right\|} }
                               {\sum_{\ell=1}^P{\left\| plog^{-1}\left(M_p(\vec{m}_{\ell})\right)-f(\vec{m}_{\ell})\right\|} }
                  \right\}	\]`

is more robust because it is based on many points :math:`\vec{m}_{\ell}`. These midpoints are usually not close to
trained points and if a single midpoint :math:`\vec{m}_{\ell}` happens to be close to trained points (and has small
errors), then this will not change the new error ratio :math:`p_k` very much, because numerator and denominator are
now **sums** of errors. The calculation of :math:`p_\text{eff}` from  :math:`p_k` proceeds in the same way as described
in :ref:`The p-Effect <pEffect-label>`.


The calculation of the new ratio :math:`p_k` and the emerging  :math:`p_\text{eff}` is done in :meth:`.Surrogator2.calcPEffectNew`.

A third option is ``ISA.onlinePLOG =`` **O_LOGIC.NONE** (no online plog).
In this case we do **not** decide online about :math:`plog()` but instead make a decision once after initial design and keep this decision for all iterations. The
decision is based on the min-max range of ``Fres`` after initial design, which is compared with
threshold ``ISA.TFRange``. If larger than threshold, apply :math:`plog()` to :math:`f()`; if smaller than threshold, use
the unsquashed :math:`f()`.



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

.. autoclass:: surrogator1.Surrogator1.AdFitter
   :members: __init__, __call__

.. autoclass:: surrogator1.Surrogator1
   :members: calcPEffect, trainSurrogates

.. autoclass:: surrogator2.Surrogator2
   :members: calcPEffectNew, trainSurrogates

.. autoclass:: rbfModel.RBFmodel
   :members: __init__, __call__

.. autoclass:: rbfSacobra.RBFsacob
   :members: __init__, __call__

.. autoclass:: evaluatorReal.EvaluatorReal
   :members: update, equ_refine, equ_num_max_viol, ine_num_max_viol
