import numpy as np


class ISAoptions:
    """
        Internal self-adjusting options for SACOBRA.

        Defaults for DOSAC=1. Other options are DOSAC=0 (ISAoptions0, COBRA-R settings = turn off SACOBRA),
        and DOSAC = 2 (ISAoptions2, SACOBRA settings with fewer parameters and more online adjustments.
    """
    def __init__(self,
                 DOSAC = 1,
                 RS=True,
                 RStype="SIGMOID",  # "CONSTANT",  "SIGMOID"
                 RSauto=False,
                 RSmax=0.3,  # maximum probability of a random start
                 RSmin=0.05,  # minimum probability of a random start
                 aDRC=True,
                 aFF=True,
                 TFRange=1e+05,
                 aCF=True,
                 TGR=1e+03,
                 conPLOG=False,
                 conFitPLOG=False,
                 adaptivePLOG=False,
                 onlinePLOG=False,
                 onlineFreqPLOG=10,
                 pEffectInit=0,
                 minMaxNormal=False,
                 onlineMinMax=False,
                 Cs=10
                 ):
        self.DOSAC = DOSAC
        self.RS = RS
        self.RStype = RStype
        self.RSauto = RSauto
        self.RSmax = RSmax
        self.RSmin = RSmin
        self.aDRC = aDRC
        self.aFF = aFF
        self.TFRange = TFRange
        self.aCF = aCF
        self. TGR = TGR
        self.conPLOG = conPLOG
        self.conFitPLOG = conFitPLOG
        self.adaptivePLOG = adaptivePLOG
        self.onlinePLOG = onlinePLOG
        self.onlineFreqPLOG = onlineFreqPLOG
        self.pEffectInit = pEffectInit
        self.minMaxNormal = minMaxNormal
        self.onlineMinMax = onlineMinMax
        self.Cs = Cs


class ISAoptions0(ISAoptions):
    def __init__(self,
                 DOSAC = 0,
                 RS=False,
                 RStype="SIGMOID",  # "CONSTANT",  "SIGMOID"
                 RSauto=False,
                 RSmax=0.3,  # maximum probability of a random start
                 RSmin=0.05,  # minimum probability of a random start
                 aDRC=False,
                 aFF=False,
                 TFRange=np.inf,
                 aCF=False,
                 TGR=np.inf,
                 conPLOG=False,
                 conFitPLOG=False,
                 adaptivePLOG=False,
                 onlinePLOG=False,
                 onlineFreqPLOG=10,
                 pEffectInit=0,
                 minMaxNormal=False,
                 onlineMinMax=False,
                 Cs=10
                 ):
        self.DOSAC = DOSAC
        self.RS = RS
        self.RStype = RStype
        self.RSauto = RSauto
        self.RSmax = RSmax
        self.RSmin = RSmin
        self.aDRC = aDRC
        self.aFF = aFF
        self.TFRange = TFRange
        self.aCF = aCF
        self. TGR = TGR
        self.conPLOG = conPLOG
        self.conFitPLOG = conFitPLOG
        self.adaptivePLOG = adaptivePLOG
        self.onlinePLOG = onlinePLOG
        self.onlineFreqPLOG = onlineFreqPLOG
        self.pEffectInit = pEffectInit
        self.minMaxNormal = minMaxNormal
        self.onlineMinMax = onlineMinMax
        self.Cs = Cs



class ISAoptions2:
    def __init__(self,
                 DOSAC=2,
                 RS=True,
                 RStype="CONSTANT",  # "CONSTANT",  "SIGMOID"
                 RSauto=True,
                 RSmax=0.3,  # maximum probability of a random start
                 RSmin=0.05,  # minimum probability of a random start
                 aDRC=True,
                 aFF=True,
                 TFRange=-1,
                 aCF=True,
                 TGR=-1,
                 conPLOG=False,
                 conFitPLOG=False,
                 adaptivePLOG=False,
                 onlinePLOG=True,
                 onlineFreqPLOG=10,
                 pEffectInit=3,
                 minMaxNormal=False,
                 onlineMinMax=False,
                 Cs=10
                 ):
        self.DOSAC = DOSAC
        self.RS = RS
        self.RStype = RStype
        self.RSauto = RSauto
        self.RSmax = RSmax
        self.RSmin = RSmin
        self.aDRC = aDRC
        self.aFF = aFF
        self.TFRange = TFRange
        self.aCF = aCF
        self. TGR = TGR
        self.conPLOG = conPLOG
        self.conFitPLOG = conFitPLOG
        self.adaptivePLOG = adaptivePLOG
        self.onlinePLOG = onlinePLOG
        self.onlineFreqPLOG = onlineFreqPLOG
        self.pEffectInit = pEffectInit
        self.minMaxNormal = minMaxNormal
        self.onlineMinMax = onlineMinMax
        self.Cs = Cs



