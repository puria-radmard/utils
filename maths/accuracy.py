def accuracy(input_sequence, target_sequence):
    assert target_sequence.shape == input_sequence.shape
    return (input_sequence == target_sequence).sum() / target_sequence.numel()


def active_accuracy(input_sequence, target_sequence, active_mask):
    assert target_sequence.shape == input_sequence.shape == active_mask.shape
    return ((input_sequence == target_sequence) * active_mask).sum() / active_mask.sum()
