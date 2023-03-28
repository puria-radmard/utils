"""
These are for neurogym tasks that are 'non-interactive', i.e. the environment object
can just be unloaded as a flat sequence with which we can do supervised learning.

This cuts out the need to loop actions and observations in the training loop, like we did before

This also, on a case-by-case basis, adjusts signals, e.g.
    for the variaous perceptual decision making tasks, the gt would activate one sample AFTER
    fixation signal inactive, rather than on the signal

Also by a case-by-case basis, we have to get the masks for each training type,
    e.g. a mask for active only training, a mask for full sequence training, 
         a mask for fixation weighted training, etc.
"""

import torch
import neurogym as ngym

from tasks.ngym.tuned_neurogym_tasks import task_directory

def full_task_and_mask_extract(ngym_env_method_name, training_type, batch_size, num_steps=100):
    observations, gts, masks = [], [], []
    for _ in range(batch_size):
        ngym_env = task_directory(ngym_env_method_name)()
        predata = run_task_fully(ngym_env, num_steps=num_steps)
        new_data = idiosyncratically_modify_data(predata, ngym_env_method_name) # Dict of obs and gts
        observations.append(new_data['obs'])
        gts.append(new_data['gt'])
        # Mask might need both obs and gts
        masks.append(idiosyncratically_generate_training_mask(new_data, ngym_env_method_name, training_type))
    masks = torch.stack(masks, 0).float() # [batch, steps]
    observations = torch.stack(observations, 0).float() # [batch, steps, obsdim]
    gts = torch.stack(gts, 0) # [batch, steps]
    return observations, gts, masks


def run_task_fully(ngym_env, num_steps=100):
    data = ngym.utils.plotting.run_env(env=ngym_env, num_steps=num_steps, def_act=None, model=None)
    return {'obs': torch.tensor(data['ob']), 'gt': torch.tensor(data['gt'])}


def idiosyncratically_modify_data(ngym_data, env_name):
    """
    Small modifications we make to each task, based on its name in ngym
    """
    # Shift gt back one to match observations - gt != 0 only when fixation = 0
    if env_name in ['PerceptualDecisionMaking', 'PulseDecisionMaking', 'PerceptualDecisionMakingDelayResponse']:
        new_gt = torch.roll(ngym_data['gt'], -1)
        assert (ngym_data['obs'][:,0].int() @ new_gt.int()) == 0.0
        return {'obs': ngym_data['obs'], 'gt': new_gt}
    else:
        raise NotImplementedError


def idiosyncratically_generate_training_mask(ngym_data, env_name, training_type):
    """
        active = only when fixation is off, typically only one sample for each sequence.
        average = flat sum here, i.e. return 1s only (averaged later on)
    """
    if training_type == 'average':
        return torch.ones_like(ngym_data['gt']).float()
    if env_name in ['PerceptualDecisionMaking', 'PulseDecisionMaking', 'PerceptualDecisionMakingDelayResponse']:
        if training_type == 'active':
            mask = torch.zeros_like(ngym_data['gt']).float()
            mask[ngym_data['gt'] != 0] = 1.
            return mask




