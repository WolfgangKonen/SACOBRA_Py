from opt.equOptions import EQUoptions
from opt.idOptions import IDoptions
from opt.isaOptions import ISAoptions
from opt.msOptions import MSoptions
from opt.rbfOptions import RBFoptions
from opt.seqOptions import SEQoptions
from opt.trOptions import TRoptions


class SACoptions:
    """
        The collection of all parameters (options) for SACOBRA, hierarchical
        organized in sub-option-classes
    """
    def __init__(self,
                 feval=100,
                 XI=None,
                 skipPhaseI=True,
                 ID=IDoptions(),
                 RBF=RBFoptions(),
                 SEQ=SEQoptions(),
                 # repairInfeas=False, ri=defaultRI(),
                 MS=MSoptions(),
                 EQU=EQUoptions(),
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
                 finalEpsXiZero=False,  # if True, then set EPS=XI=0 in final iteration (full exploit, might require
                                        # SEQ.conTol=1e-7 instead of 0.0)
                 verbose=1, verboseIter=10, important=True,
                 cobraSeed=42
                 ):

        self.feval = feval
        self.skipPhaseI = skipPhaseI
        self.ID = ID
        self.epsilonInit = epsilonInit
        self.epsilonMax = epsilonMax
        self.finalEpsXiZero = finalEpsXiZero
        self.penaF = penaF
        self.sigmaD = sigmaD
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
