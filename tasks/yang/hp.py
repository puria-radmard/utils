"""
Get the default hyperparameters for each task. Taken from train.py in gyyang repository
"""
from tasks.yang.task import get_num_ring, get_num_rule, rules_dict
import random
import numpy as np


def get_default_hp(ruleset, task_name):
    '''Get a default hp.
    Useful for debugging.
    Returns:
        hp : a dictionary containing training hpuration
    '''

    num_ring = get_num_ring(ruleset)
    n_rule = get_num_rule(ruleset)

    n_eachring = 32
    n_input, n_output = 1+num_ring*n_eachring+n_rule, n_eachring+1
    hp = {
            # batch size for training
            'batch_size_train': 64,
            # batch_size for testing
            'batch_size_test': 512,
            # input type: normal, multi
            'in_type': 'normal',
            # Type of loss functions
            'loss_type': 'lsq',
            # Time constant (ms)
            # 'tau': 100,                 # XXX: keep this the same?
            # discretization time step (ms)
            'dt': 5, # 20,              # XXX: changed this!
            # # recurrent noise
            # 'sigma_rec': 0.05,
            # input noise
            'sigma_x': 0.00,       # No noise for now!
            # proportion of weights to train, None or float between (0, 1)
            # 'p_weight_train': None,
            # # Stopping performance
            # 'target_perf': 1.,
            # number of units each ring
            'n_eachring': n_eachring,
            # # number of rings
            # 'num_ring': num_ring,
            # # number of rules
            # 'n_rule': n_rule,
            # first input index for rule units
            'rule_start': 1+num_ring*n_eachring,
            # number of input units
            'n_input': n_input,
            # number of output units
            'n_output': n_output,
            # # number of recurrent units
            # 'n_rnn': 256,
            # number of input units
            'ruleset': ruleset,
            # # intelligent synapses parameters, tuple (c, ksi)
            # 'c_intsyn': 0,
            # 'ksi_intsyn': 0,
            }

    # hp['alpha'] = hp['dt']/hp['tau']

    hp['rule_trains'] = rules_dict[ruleset]

    seed = random.randint(0, 1000)
    hp['seed'] = seed
    hp['rng'] = np.random.RandomState(seed)

    return hp
