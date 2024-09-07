import torch

def accuracy(input_sequence, target_sequence):
    assert target_sequence.shape == input_sequence.shape
    return (input_sequence == target_sequence).sum() / target_sequence.numel()


def active_accuracy(input_sequence, target_sequence, active_mask):
    assert target_sequence.shape == input_sequence.shape == active_mask.shape
    return ((input_sequence == target_sequence) * active_mask).sum() / active_mask.sum()

def temporal_majority_accuracy(rnn_readout, labels):
    """
    Gets a single estimate for each instance by getting 
        the modal estimate across time

    rnn_readout: [B, T, C, (trials)]
    labels: [B]
    """
    assert labels.max() < rnn_readout.shape[2]
    temporal_estimates = rnn_readout.argmax(2)                   # [B, T, (trials)]
    modal_estimate = torch.mode(temporal_estimates, 1).values   # [B, (trials)]
    if len(modal_estimate.shape) == 2:
        labels = labels.unsqueeze(-1)
    return (modal_estimate == labels).float().mean()


def temporal_majority_accuracy_from_saccade(rnn_readout, labels, num_classes):
    """
    Gets a single estimate for each instance by getting the modal estimate across time
    This time with saccade directions, assumed to be even around the unit circle

    rnn_readout: [B, T, 2 (x, y), (trials)]
    labels: [B]
    """
    incl_trials = len(rnn_readout.shape) == 4
    if not incl_trials:
        rnn_readout = rnn_readout.unsqueeze(-1)
    else:
        raise NotImplementedError('return (...).reshape(B, -1) below!')
    B, T, _, trials = rnn_readout.shape
    num_reps = B * T * trials

    perm = [2, 3, 1, 0] # put the xy at the end then flip
    reshaped_rnn_readout = rnn_readout.permute(*perm)             # [xy, trials, T, batch]
    angles = torch.pi + torch.arctan2(*reshaped_rnn_readout).T      # [batch, T, trials]
    incre = (2 * torch.pi / num_classes)
    preds = torch.div(angles, incre, rounding_mode='trunc').reshape(B, -1)

    modal_estimate = torch.mode(preds, 1).values
    
    return (modal_estimate == labels.to(modal_estimate.device)).float().mean()

