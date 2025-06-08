------------
Appendix
------------

This chapter has details on some of **SACOBRA_Py**'s data structures.

With

.. code-block::

          cobra = CobraInitializer(...)

we generate the main data structure ``cobra`` of **SACOBRA_Py**. Some of its data frame and dictionary elements are described below.

.. _sacres-label:

cobra.sac_res
-------------

``cobra.sac_res`` is a dictionary which holds **SACOBRA_Py**'s **results** from the initialization phase and from the optimization phase.

.. .. autodata:: cobraInit.CobraInitializer.sac_res
   <this seems not to work so far>

.. autofunction:: cobraInit.CobraInitializer.create_sac_res

.. hint:: some text


.. _df-label:

cobra.df
---------

``cobra.df`` is a data frame which holds diagnostic information about the optimization process.
``df`` is only for diagnostics, its elements are not used by the optimization process in any form.

.. autofunction:: cobraPhaseII.CobraPhaseII.create_df



.. _df2-label:

cobra.df2
---------

``cobra.df2`` is a data frame which holds diagnostic information about the optimization process.
``df2`` is only for diagnostics, its elements are not used by the optimization process in any form.

.. autofunction:: cobraPhaseII.CobraPhaseII.create_df2
