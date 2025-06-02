from opt.equOptions import EQUoptions
from opt.idOptions import IDoptions
from opt.isaOptions import ISAoptions
from opt.msOptions import MSoptions
from opt.rbfOptions import RBFoptions
from opt.seqOptions import SEQoptions
from opt.trOptions import TRoptions


class SACoptions:
    """
        The collection of all parameters (options) for **SACOBRA_Py**. Except for some general
        parameters defined in this class, they are hierarchically organized in nested option classes.

        :param feval: number of function evaluations
        :param XI:  Distance-Requirement-Cycle (:ref:`DRC <DRC-label>`) that controls exploration: Each infill point has a forbidden-sphere of radius ``XI[c]`` around it. ``c`` loops cyclically through ``XI``'s inidices. If ``XI==None``, then CobraInitializer will set it, depending on objective range, to short DRC ``[0.001, 0.0]`` or long DRC ``[0.3, 0.05, 0.001, 0.0005, 0.0]``.
        :param skipPhaseI: whether to skip **SACOBRA_Py** phase I or not
        :param DOSAC: controls the default options for ``ISAoptions ISA``. 0: take plain COBRA settings, 1: full SACOBRA settings, 2: reduced SACOBRA settings
        :param saveIntermediate: whether to save intermediate results or not (TODO)
        :param saveSurrogates: whether to save surrogate models or not (TODO)
        :param verbose: verbosity level: 0: print nothing. 1: print only important messages. 2: print everything
        :param verboseIter: an integer value, after how many iterations to print summarized results.
        :param important: controls the importance level for some ``verboseprint``'s in ``updateInfoAndCounters``
        :param cobraSeed: the seed for RNGs. **SACOBRA_Py** guarantees the same results for the same seed
        :param ID: nested options for initial design
        :type ID: IDoptions
        :param RBF: nested options for radial basis functions
        :type RBF: RBFoptions
        :param SEQ: nested options for sequential optimizer
        :type SEQ: SEQoptions
        :param EQU: nested options for equality constraints
        :type EQU: EQUoptions
        :param ISA: nested Internal SACOBRA options
        :type ISA: ISAoptions
        :param MS: nested options for model selection
        :type MS: MSoptions
        :param TR: nested options for trust region
        :type TR: TRoptions
    """
    def __init__(self,
                 feval=100,
                 XI=None,
                 skipPhaseI=True,
                 # DOSAC=1,
                 saveIntermediate=False,
                 saveSurrogates=False,
                 verbose=1, verboseIter=10, important=True,
                 cobraSeed=42,
                 ID=IDoptions(),
                 RBF=RBFoptions(),
                 SEQ=SEQoptions(),
                 EQU=EQUoptions(),
                 ISA=ISAoptions(),
                 # repairInfeas=False, ri=defaultRI(),
                 MS=MSoptions(),
                 TR=TRoptions(),
                 # conditioningAnalysis=defaultCA(),
                 # constraintHandling="DEFAULT",
                 # DEBUG_RBF=defaultDebugRBF(), DEBUG_TR=False,
                 # DEBUG_TRU=False, DEBUG_RS=False, DEBUG_XI=False,
                 ):
        """
        """
        self.feval = feval
        self.XI = XI
        self.skipPhaseI = skipPhaseI
        # self.DOSAC = DOSAC
        self.saveIntermediate = saveIntermediate
        self.saveSurrogates = saveSurrogates
        self.verbose = verbose
        self.verboseIter = verboseIter
        self.important = important
        self.cobraSeed = cobraSeed
        self.ID = ID
        self.RBF = RBF
        self.SEQ = SEQ
        self.EQU = EQU
        self.ISA = ISA
        self.MS = MS
        self.TR = TR
