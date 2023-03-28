import torch


# Task construction - remove from this file
# Rather than a Wfix (size = N x 1) and a Winp * exc_only_mask (size = 8 x N x 1),
# I will use a single W_input * exc_only_mask (size = N x 9)
# and an input stream of 9 dimensions, with first one being a fixation binary, and the others being direction binaries


def generate_complete_wm_task(N_pat, stim_steps, mult):

    # NB: all time will have to be multiplied by stim_steps
    batch_size = N_pat * mult
    input = torch.zeros([batch_size, 6*stim_steps, N_pat + 1]).to('cuda')  # Batch size 8, two for each possible task
    input[:, : 5*stim_steps, 0] = 1.0                              # Keep fixation off at end only
    for i in range(N_pat):
        for j in range(mult):
            input[i + j * N_pat, 2*stim_steps: 4*stim_steps, i + 1] = 1.0        # Mult sequences for each possible task


    target = torch.zeros([batch_size, 6*stim_steps]).long()
    for i in range(N_pat):
        for j in range(mult):
            target[i + j * N_pat, 5*stim_steps:] = i

    return (input, target)



def generate_wm_task(N_pat, stim_steps, mult):

    # NB: all time will have to be multiplied by stim_steps
    batch_size = N_pat * mult
    input = torch.zeros([batch_size, 6, N_pat + 1]).to('cuda')  # Batch size 8, two for each possible task
    input[:, : 5, 0] = 1.0                              # Keep fixation off at end only
    for i in range(N_pat):
        for j in range(mult):
            input[i + j * N_pat, 2, i + 1] = 1.0        # Mult sequences for each possible task


    res_target = torch.zeros([batch_size, stim_steps]).long()
    dly_target = torch.zeros([batch_size, 2*stim_steps]).long()
    for i in range(N_pat):
        for j in range(mult):
            dly_target[i + j * N_pat,:] = i
            res_target[i + j * N_pat,:] = i

    return (
        input,
        res_target,#.reshape(stim_steps, N_pat, 1),
        dly_target,#.reshape(stim_steps*2, N_pat, 1)
    )
