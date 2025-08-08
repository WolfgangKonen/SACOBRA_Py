import numpy as np
from enum import Enum


class RSTYPE(Enum):
    SIGMOID = 1
    CONSTANT = 2


class O_LOGIC(Enum):
    NONE = 0
    XNEW = 1
    MIDPTS = 2


class ISAoptions:
    """
        Internal self-adjusting (ISA) options for SACOBRA.

        Defaults for ``isa_ver=1`` (SACOBRA settings with fewer online adjustments). Other choices are

        - ``isa_ver=0`` (:class:`.ISAoptions0`, COBRA-R settings = turn off SACOBRA), and
        - ``isa_ver=2`` (:class:`.ISAoptions2`, SACOBRA settings with fewer parameters and more online adjustments).

        :param isa_ver: ISA version number (was ``DOSAC`` in R)
        :param RS: flag to enable/disable random start algorithm
        :param RStype: type of function to calculate probability to start the internal optimizer with a random start
                       point. One out of [RSTYPE.CONSTANT, RSTYPE.SIGMOID]
        :param RSauto:
        :param RSmax: maximum probability of a random start
        :param RSmin: minimum probability of a random start
        :param RS_Cs: if RS_Cs iterations w/o progress, do a random start
        :param RS_rep: if True, generate reproducible random numbers with my_rng2 (R and Python)
        :param RS_verb: if True, be verbose in RandomStarter.random_start
        :param aDRC: flag for automatic DRC adjustment
        :param aFF: flag for automatic objective function transformation
        :param aCF: flag for automatic constraint function transformation
        :param TFRange: threshold, if the range of ``Fres`` is larger than ``TFRange`` and if ``onlinePLOG=NONE``,
                    then apply plog to ``Fres`` (objective function values)
        :param TGR: threshold: If ``GRatio > TGR``, then apply automatic constraint function transformation. ``GRatio``
                    is the ratio "largest GR / smallest GR" where GR is the min-max range of a specific constraint.
                    If ``TGR < 1``, then the transformation is always performed.
        :param conPLOG:
        :param conFitPLOG:
        :param adaptivePLOG: (experimental) flag for objective function transformation with ``plog``, where the
                             parameter ``pShift`` is adapted during iterations
        :param onlinePLOG: three-valued logic for online decision-making: O_LOGIC.NONE (no online, only fixed initial
                    decision whether to use plog); O_LOGIC.XNEW (online plog decision according to pEffect and
                    pEffect-calculation is based on new infill point ``xNew``); O_LOGIC.MIDPTS (online plog decision
                    according to pEffect and pEffect-calculation is based on midpoints)
        :param onlineFreqPLOG: after how many iterations the online plog check is done again
        :param pEffectInit: the initial value for ``pEffect``, needed for first pass through cobraPhaseII while loop in
                    case ``O_LOGIC.XNEW``. Not needed in cases  ``O_LOGIC.NONE`` or ``O_LOGIC.MIDPTS``
        :param pEff_npts: the number p of initial design points to take from initial design matrix ``A`` to form
                    p*(p-1)/2 pairs and calculate pair midpoints for case ``O_LOGIC.MIDPTS``
        :param minMaxNormal:
        :param onlineMinMax:
    """
    # obsolete:   :param pEffectLogic: logic for pEffect calculation. One out of [O_LOGIC.XNEW, O_LOGIC.MIDPTS]
    def __init__(self,
                 isa_ver=1,
                 RS=True,
                 RStype=RSTYPE.CONSTANT,  # .CONSTANT or .SIGMOID
                 RSauto=False,
                 RSmax=0.3,  #
                 RSmin=0.05,  #
                 RS_Cs=10,  #
                 RS_rep=False,  #
                 RS_verb=False,  #
                 aDRC=True,
                 aFF=True,
                 aCF=True,
                 TFRange=1e+05,
                 TGR=1e+03,
                 conPLOG=False,
                 conFitPLOG=False,
                 adaptivePLOG=False,
                 onlinePLOG=O_LOGIC.NONE,
                 onlineFreqPLOG=10,
                 pEffectInit=0,
                 pEff_npts=3,
                 pEff_DBG=False,
                 minMaxNormal=False,
                 onlineMinMax=False
                 ):
        self.isa_ver = isa_ver
        self.RS = RS
        self.RStype = RStype
        self.RSauto = RSauto
        self.RSmax = RSmax
        self.RSmin = RSmin
        self.RS_Cs = RS_Cs
        self.RS_rep = RS_rep
        self.RS_verb = RS_verb
        self.aDRC = aDRC
        self.aFF = aFF
        self.aCF = aCF
        self.TFRange = TFRange
        self.TGR = TGR
        self.conPLOG = conPLOG
        self.conFitPLOG = conFitPLOG
        self.adaptivePLOG = adaptivePLOG
        self.onlinePLOG = onlinePLOG
        self.onlineFreqPLOG = onlineFreqPLOG
        self.pEffectInit = pEffectInit
        self.pEff_npts = pEff_npts
        self.pEff_DBG = pEff_DBG
        self.minMaxNormal = minMaxNormal
        self.onlineMinMax = onlineMinMax


class ISAoptions0(ISAoptions):
    """
        Internal self-adjusting options for ``isa_ver=0`` (COBRA-R settings = turn off SACOBRA).

        Differs only in default settings. The parameters and their meaning are the same as in :class:`.ISAoptions`.
    """
    def __init__(self,
                 isa_ver=0,
                 RS=False,
                 RStype=RSTYPE.CONSTANT,  # .CONSTANT or .SIGMOID
                 RSauto=False,
                 RSmax=0.3,  # maximum probability of a random start
                 RSmin=0.05,  # minimum probability of a random start
                 RS_Cs=10,  # if RS_Cs iterations w/o progress, do a random start
                 RS_rep=False,  # if True, generate reproducible random numbers with my_rng2 (R and Python)
                 RS_verb=False,  # if True, be verbose in RandomStarter.random_start
                 aDRC=False,
                 aFF=False,
                 aCF=False,
                 TFRange=np.inf,
                 TGR=np.inf,
                 conPLOG=False,
                 conFitPLOG=False,
                 adaptivePLOG=False,
                 onlinePLOG=O_LOGIC.NONE,
                 onlineFreqPLOG=10,
                 pEffectInit=0,
                 pEff_npts=3,
                 pEff_DBG=False,
                 minMaxNormal=False,
                 onlineMinMax=False
                 ):
        super().__init__(
            isa_ver=isa_ver,
            RS=RS,
            RStype=RStype,
            RSauto=RSauto,
            RSmax=RSmax,
            RSmin=RSmin,
            RS_Cs=RS_Cs,
            RS_rep=RS_rep,
            RS_verb=RS_verb,
            aDRC=aDRC,
            aFF=aFF,
            aCF=aCF,
            TFRange=TFRange,
            TGR=TGR,
            conPLOG=conPLOG,
            conFitPLOG=conFitPLOG,
            adaptivePLOG=adaptivePLOG,
            onlinePLOG=onlinePLOG,
            onlineFreqPLOG=onlineFreqPLOG,
            pEffectInit=pEffectInit,
            pEff_npts=pEff_npts,
            pEff_DBG=pEff_DBG,
            minMaxNormal=minMaxNormal,
            onlineMinMax=onlineMinMax
        )


class ISAoptions2(ISAoptions):
    """
        Internal self-adjusting options for ``isa_ver=2`` (SACOBRA with fewer parameters and more online adjustments).

        Differs only in default settings. The parameters and their meaning are the same as in :class:`.ISAoptions`.
    """
    def __init__(self,
                 isa_ver=2,
                 RS=True,
                 RStype=RSTYPE.CONSTANT,  # .CONSTANT or .SIGMOID
                 RSauto=True,
                 RSmax=0.3,  # maximum probability of a random start
                 RSmin=0.05,  # minimum probability of a random start
                 RS_Cs=10,  # if RS_Cs iterations w/o progress, do a random start
                 RS_rep=False,  # if True, generate reproducible random numbers with my_rng2 (R and Python)
                 RS_verb=False,  # if True, be verbose in RandomStarter.random_start
                 aDRC=True,
                 aFF=True,
                 aCF=True,
                 TFRange=-1,
                 TGR=-1,
                 conPLOG=False,
                 conFitPLOG=False,
                 adaptivePLOG=False,
                 onlinePLOG=O_LOGIC.NONE,  # TODO: there is a bug with onlinePLOG and ISAoptions2
                 onlineFreqPLOG=10,
                 pEffectInit=3,
                 pEff_npts=3,
                 pEff_DBG=False,
                 minMaxNormal=False,
                 onlineMinMax=False
                 ):
        super().__init__(
            isa_ver=isa_ver,
            RS=RS,
            RStype=RStype,
            RSauto=RSauto,
            RSmax=RSmax,
            RSmin=RSmin,
            RS_Cs=RS_Cs,
            RS_rep=RS_rep,
            RS_verb=RS_verb,
            aDRC=aDRC,
            aFF=aFF,
            aCF=aCF,
            TFRange=TFRange,
            TGR=TGR,
            conPLOG=conPLOG,
            conFitPLOG=conFitPLOG,
            adaptivePLOG=adaptivePLOG,
            onlinePLOG=onlinePLOG,
            onlineFreqPLOG=onlineFreqPLOG,
            pEffectInit=pEffectInit,
            pEff_npts=pEff_npts,
            pEff_DBG=pEff_DBG,
            minMaxNormal=minMaxNormal,
            onlineMinMax=onlineMinMax
        )
