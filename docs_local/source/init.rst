--------------
Initialization
--------------

This chapter describes the SACOBRA initialization and the options (or hyperparameters) of SACOBRA.



CobraInitializer
----------------

The initialization in **SACOBRA_Py** is done by creating an object of class :class:`.CobraInitializer` and it consists of the following steps:

- pass in the specification of the COP
- pass in all the options in :class:`.SACoptions` ``s_opt``
- (optional, if ``s_opts.ID.rescale==True``) rescale the problem in input space
- create the initial design, see :class:`.InitDesigner`
- adjust several elements according to constraint range, see :meth:`.CobraInitializer.adCon`
- calculate for each initial design point ``numViol``, the number of violated constraints, and ``maxViol``, the maximum constraint violation. If equality constraints are involved, calculate :math:`\mu_{init}`, the radius for an artificial feasibility tube around each equality constraint (see :class:`.EQUoptions`) and base the calculation of ``numViol`` and ``maxViol`` on this artificial feasibility. 
- calculate the so-far best (artificial) feasible point. If no point fulfills (artificial) feasibility, form the set of points with minimum ``numViol`` and take from this set the one point with the best objective.
- set up result dictionary :ref:`sac_res <sacres-label>`
- adjust :ref:`DRC <DRC-label>` according to objective range, see :meth:`.adDRC`


.. autoclass:: cobraInit.CobraInitializer 
   :members: adCon, adDRC, get_fbest, get_xbest, get_xbest_cobra

.. autoclass:: initDesigner.InitDesigner
   :members: __call__

TODO: Describe **Types of Initial Design** (LHS, Random, ...)


Options
-------

All paramters (options) for SACOBRA_Py have sensible defaults defined. The user has only to specify those parameters where a 
setting different from the defaults is desired.

.. autoclass:: opt.sacOptions.SACoptions


.. _DRC-label:

Distance Requirement Cycle (DRC)
================================

The Distance Requirement Cycle (DRC) is the vector ``XI`` that controls exploration: Each already evaluated infill point is surrounded by a forbidden-sphere of radius ``XI[c]`` with ``c = i mod XI.size`` (``c`` loops cyclically through ``XI``’s inidices`, that's where the name *cycle* comes from). A new infill point is searched under the additional constraint that it has to be a distance ``XI[c]`` away from all other already evaluted infill points. The larger ``XI[c]``, the more exploration.

If ``XI==None``, then :class:`.CobraInitializer` will set it, depending on objective range, to short DRC ``[0.001, 0.0]`` or long DRC ``[0.3, 0.05, 0.001, 0.0005, 0.0]``. Both vectors contain ``XI[c] = 0`` which enforces exploitation. (If all entries were ``XI[c] > 0`` then a good region would never be exploited further, since the search could never continue in the close vicinity of an already good infill point.)

.. autoclass:: opt.idOptions.IDoptions

.. autoclass:: opt.rbfOptions.RBFoptions

.. autoclass:: opt.seqOptions.SEQoptions

.. autoclass:: opt.equOptions.EQUoptions

TODO: Describe **Refine Algo**

.. autoclass:: opt.isaOptions.ISAoptions

.. autoclass:: opt.isaOptions.ISAoptions0

.. autoclass:: opt.isaOptions.ISAoptions2



