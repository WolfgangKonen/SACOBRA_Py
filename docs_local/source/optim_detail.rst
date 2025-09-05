--------------------
Details Optimization
--------------------


Details Phase II
-----------------

.. autoclass:: randomStarter.RandomStarter
   :members: __init__, random_start

.. autoclass:: seqOptimizer.SeqOptimizer

.. _AdFitter-label:

.. autoclass:: surrogator1.Surrogator1.AdFitter
   :members: __init__, __call__

.. autoclass:: surrogator1.Surrogator1
   :members: calcPEffect, trainSurrogates

.. autoclass:: surrogator2.Surrogator2
   :members: calcPEffect, trainSurrogates

.. autoclass:: rbfModel.RBFmodel
   :members: __init__, __call__

.. autoclass:: rbfSacobra.RBFsacob
   :members: __init__, __call__

.. autoclass:: evaluatorReal.EvaluatorReal
   :members: update, equ_refine, equ_num_max_viol, ine_num_max_viol
