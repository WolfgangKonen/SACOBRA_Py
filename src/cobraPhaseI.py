import numpy as np
# need to specify full path here for test units to run smoothly:
from cobraInit import CobraInitializer


class CobraPhaseI:
    def __init__(self, cobra: CobraInitializer):

        #
        # STEP 0: first settings and checks
        #
        s_opts = cobra.get_sac_opts()
        s_res = cobra.get_sac_res()

        #
        # STEP 4: update structures
        #
        self.phase = "phase1"
        self.sac_opts = s_opts
        self.sac_res = s_res

    def get_sac_opts(self):
        return self.sac_opts

    def get_sac_res(self):
        return self.sac_res

    def __call__(self, *args, **kwargs):
        return self.sac_opts, self.sac_res