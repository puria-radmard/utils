"""
Sometimes, WM tasks can get too hard, and the long delay between the cue and the action time
    can cause a greatly diminishing gradient.

This leads me to think that these require a training aid, like the one Wanye provided the simple
    8-point WM task. In that case, the targets during post-cue-fixation and actual action time
    were the same, and the only difference was the set of readout weights that were used to decode
    the SSN rates
"""

import torch

def get_delay_time_bounds():

    pass
