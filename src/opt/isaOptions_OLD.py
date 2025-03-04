import numpy as np


class ISAoptions:
    """
        Internal self-adjusting options for SACOBRA
    """

    def __init__(self, DOSAC=1):
        """
            We pass in only parameter DOSAC, the other parameters are set to defaults, depending on DOSAC.
            If we want other values for these parameters, we have to set them after the ISAoptions object
            is constructed.
        :param DOSAC: 0,1 or 2 [default: 1].
            0: COBRA-R settings (turn off SACOBRA),
            1: SACOBRA settings,
            2: SACOBRA settings with fewer parameters and more online adjustments (aFF and aCF are done parameter free).
        """
        if DOSAC == 0:
            s = ISAoptions0()
        elif DOSAC == 1:
            s = ISAoptions1()
        elif DOSAC == 2:
            s = ISAoptions2()
        else:
            raise RuntimeError(f"Option DOSAC = {DOSAC} is not allowed (has to be 0,1, or 2)")

        self.DOSAC = DOSAC
        self.RS = s.RS
        self.RStype = s.RStype
        self.RSauto = s.RSauto
        self.RSmax = s.RSmax
        self.RSmin = s.RSmin
        self.aDRC = s.aDRC
        self.aFF = s.aFF
        self.TFRange = s.TFRange
        self.aCF = s.aCF
        self.TGR = s.TGR
        self.conPLOG = s.conPLOG
        self.conFitPLOG = s.conFitPLOG
        self.adaptivePLOG = s.adaptivePLOG
        self.onlinePLOG = s.onlinePLOG
        self.onlineFreqPLOG = s.onlineFreqPLOG
        self.pEffectInit = s.pEffectInit
        self.minMaxNormal = s.minMaxNormal
        self.onlineMinMax = s.onlineMinMax
        self.Cs = s.Cs


class ISAoptions0(ISAoptions):
    def __init__(self,
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


class ISAoptions1:
    def __init__(self,
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


if __name__ == '__main__':
    for DOSAC in range(3):
        isa = ISAoptions(DOSAC=DOSAC)
        print(DOSAC, isa.aDRC)
    print(isa.__class__)
