import numpy as np

class FnArchiveFactory:
    """
    Build a wrapper for function ``fn`` with input and function value archive.

    Usage:

    .. code-block::

        fnArch = FnArchiveFactory(fn, x0)
        val = fnArch(x)
        #  ...
        f_arch = fnArch.getFuncArchive()

    :param fn: the function to be wrapped and archived
    :param x0: a typical input vector, needed only to infer the archive sizes

    """
    def __init__(self, fn, x0):
        self.fn = fn
        self.soluArchive = np.zeros((0,x0.size))        # an empty array with the right size for np.vstack
        self.funcArchive = np.zeros((0,fn(x0).size))    #      "     "              "     "           "

    def __call__(self, x):
        """
        Call ``v = fn(x)`` and store ``x`` and function value ``v`` in archives.
        """
        v = self.fn(x)
        self.soluArchive = np.vstack((self.soluArchive, x))
        self.funcArchive = np.vstack((self.funcArchive, v))
        return v

    def getSoluArchive(self):
        """
        :return: an array where each row is an ``x`` from a call to ``fnArch(x)``
        """
        return self.soluArchive

    def getFuncArchive(self):
        """
        :return:  an array where each row is an ``v`` from a call to ``v = fnArch(x)``
        """
        return self.funcArchive


