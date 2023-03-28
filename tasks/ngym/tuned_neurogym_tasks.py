from __future__ import annotations

"""
Lifted directly from previous repo, these are just tasks that have one decision at the end
"""

import neurogym as ngym


def task_directory(name):
    return {
        "PerceptualDecisionMaking": create_perceptual_decision_making_env,
        "PulseDecisionMaking": create_pulse_decision_making_env,
        "PerceptualDecisionMakingDelayResponse": create_perceptual_decision_making_delayed_response_env
    }[name]


# TODO: put all of this into a config file
gap = 1000

def _make_env(name, kwargs, show):
    env = ngym.make(name, **kwargs)
    if show:
        print(env)
    return env

def create_perceptual_decision_making_env(show=False):
    name = "PerceptualDecisionMaking-v0"
    kwargs = {
        "dt": 100,
        "timing": {
            "fixation": gap + 2900,
            "stimulus": 6000,
            "delay": 100,
            "decision": 100,
        },
    }
    return _make_env(name, kwargs, show)


def create_perceptual_decision_making_delayed_response_env(show=False):
    name = "PerceptualDecisionMaking-v0"
    kwargs = {
        "dt": 100,
        "timing": {
            "fixation": gap + 1000,
            "stimulus": 6000,
            "delay": 2000,
            "decision": 3000,
        },
    }
    return _make_env(name, kwargs, show)


def create_pulse_decision_making_env(show=False):
    name = "PulseDecisionMaking-v0"
    # Not sure why these are not parameterised
    n_bin = 30
    timing = {
        "fixation": gap + 3000,
        "delay": 100,
        "decision": 3000,
    }
    for i in range(n_bin):
        timing["cue" + str(i)] = 100
        timing["bin" + str(i)] = 100
    kwargs = {
        "dt": 100,
        "p_pulse": (0.5, 0.5),
        "n_bin": n_bin,
        "timing": timing,
    }
    return _make_env(name, kwargs, show)

