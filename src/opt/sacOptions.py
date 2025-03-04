import numpy as np
from opt.equOptions import EQUoptions
from opt.idOptions import IDoptions
from opt.isaOptions import ISAoptions
from opt.msOptions import MSoptions
from opt.rbfOptions import RBFoptions
from opt.seqOptions import SEQoptions
from opt.trOptions import TRoptions



class SACoptions:

    def __init__(self,
                 feval=100,
                 # initDesign="RANDOM", # "LHS",
                 # initDesPoints=None, initDesOptP=None, initBias=0.005,
                 # rescale=True, newlower=-1, newupper=1,
                 ID=IDoptions(),
                 RBF=RBFoptions(),
                 XI=None,
                 skipPhaseI=True,
                 SEQ=SEQoptions(),
                 # repairInfeas=False, ri=defaultRI(),
                 MS=MSoptions(),
                 EQU = EQUoptions(),
                 TR=TRoptions(),
                 DOSAC=1,
                 ISA=ISAoptions(),
                 # conditioningAnalysis=defaultCA(),
                 penaF=[3.0, 1.7, 3e5], sigmaD=[3.0, 2.0, 100], constraintHandling="DEFAULT",
                 # DEBUG_RBF=defaultDebugRBF(), DEBUG_TR=False,
                 DEBUG_TRU=False, DEBUG_RS=False, DEBUG_XI=False,
                 saveIntermediate=False,
                 saveSurrogates=False,
                 epsilonInit=None, epsilonMax=None,
                 verbose=1, verboseIter=10, important=True,
                 cobraSeed=42
                 ):

        self.feval = feval
        self.skipPhaseI = skipPhaseI
        self.ID = ID
        self.epsilonInit = epsilonInit
        self.epsilonMax = epsilonMax
        self.penaF = penaF
        self.DOSAC = DOSAC
        self.ISA = ISA
        self.XI = XI
        self.verbose = verbose
        self.verboseIter = verboseIter
        self.important = important
        self.cobraSeed = cobraSeed
        self.RBF = RBF
        self.SEQ = SEQ
        self.MS = MS
        self.EQU = EQU
        self.TR = TR
        self.saveIntermediate = saveIntermediate
        self.saveSurrogates = saveSurrogates
        dummy = 0
