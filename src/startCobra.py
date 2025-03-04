from innerFuncs import verboseprint
from SACOBRA_Py.src.cobraInit import CobraInitializer
from SACOBRA_Py.src.cobraPhaseI import CobraPhaseI
from SACOBRA_Py.src.cobraPhaseII import CobraPhaseII

def startCobra(cobra: CobraInitializer):
    """
    Start SACOBRA (self-adjusting constraint-based optimization) phase I and/or phase II
    for objects ``sac_opts`` and ``sac_res``

    :param cobra: object of :class:`CobraInitializer`
    :return: ``sac_res``, a (modified) object of class ``SACresults``
    """
    sac_opts = cobra.get_sac_opts()
    sac_res = cobra.get_sac_res()
    feasibleSolutionExists = (0 in sac_res['numViol'] or sac_opts.skipPhaseI)
    if feasibleSolutionExists:
        # If there is any point with no constraint violation
        cobra2 = CobraPhaseII(cobra)
        return cobra2
    else:
        verboseprint(verbose=2, important=False, message="Starting COBRA PHASE I ")
        cobra1 = CobraPhaseI(cobra)
        cobra2 = CobraPhaseII(cobra1)
        return cobra2
