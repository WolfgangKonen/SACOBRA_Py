import numpy as np

class FnArchiveFactory:
    def __init__(self, fn, x0):
        self.fn = fn
        self.soluArchive = np.zeros((0,x0.size))        # an empty array with the right size for np.vstack
        self.funcArchive = np.zeros((0,fn(x0).size))    #      "     "              "     "           "

    def __call__(self, x):
        res = self.fn(x)
        self.soluArchive = np.vstack((self.soluArchive, x))
        self.funcArchive = np.vstack((self.funcArchive, res))
        return res

    def getSoluArchive(self):
        return self.soluArchive

    def getFuncArchive(self):
        return self.funcArchive


