------------------------
Reproducible Experiments
------------------------
This chapter shows how to perform reproducible experiments with **SACOBRA_Py**.

Experiments with G-problems
---------------------------
Method :meth:`.OneS.one_s_multi_g_r` described below allows to perform reproducible experiments with detailed output.
Each run ``r`` is started with a specific seed ``cobraSeed+r`` which is saved in :ref:`dfsum <dfsum-label>`.
Each call generates a unique directory ``test/feather/run%Y-%m-%d_%Hh%Mm%S`` (year-month-day_hour-minute-second)
which contains upon successful completion the following files:

- ``box_run%Y-%m-%d.png``: summary box plot with a error box for each ``(gname,dim)``-combination
- ``dfsum.csv``: data frame :ref:`dfsum <dfsum-label>`, CSV format
- ``dfsum.feather``: data frame :ref:`dfsum <dfsum-label>`, feather format
- ``%gname_%dim_%meth_%run.png``: plot error-vs-iterations for each ``(gname,dim,meth,run)``-combination
- ``%gname_%dim_%meth_%run_df1.feather``: data frame ``df`` for this run
- ``%gname_%dim_%meth_%run_df2.feather``: data frame ``df2`` for this run
- ``%gname_%dim_%meth_sac_opts.pickle``: :class:`.SACoptions` optimization settings ``cobra.sac_opts`` for this ``(gname,dim,meth)``-combination
- ``med_std_grp.csv``: group data frame ``dfsum`` by ``(gname,dim,meth)`` with operators median and standard deviation
  and write results to new data frame with columns ``time,err,...,std_time,std_err``
- ``s_opts.pickle``: :class:`.SACoptions` optimization settings ``cobra.sac_opts`` (last run)

It is recommended that the last run uses ``meth='one_s'``, then the settings saved to ``s_opts.pickle`` are those valid
for **all** ``'one_s'`` runs.

.. autoclass:: one_s_multi_g.OneS
   :members: one_s,

.. _dfsum-label:

.. autoclass:: one_s_multi_g.OneS
   :members: one_s_multi_g_r,

