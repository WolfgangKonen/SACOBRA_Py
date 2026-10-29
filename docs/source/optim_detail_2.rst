----------------------
Details Optimization 2
----------------------


Details Phase II - Part 2
-------------------------

.. _AdFitter-label:

.. autoclass:: surrogator1.Surrogator1.AdFitter
   :members: __init__, __call__

.. autoclass:: surrogator1.Surrogator1
   :members: calcPEffect, trainSurrogates

.. autoclass:: surrogator2.Surrogator2
   :members: calcPEffect, trainSurrogates

.. It might be necessary to comment out once the line
       from surrogator import Surrogator
   in phase2Funcs.py in order to get evaluatorReal.py compiled by Sphinx. Once this is done, comment it in again.
.. autoclass:: evaluatorReal.EvaluatorReal
   :members: update, equ_refine, equ_num_max_viol, ine_num_max_viol
