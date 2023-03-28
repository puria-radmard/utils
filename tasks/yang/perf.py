import numpy as np

def popvec(y):
    """Population vector read out.
    Assuming the last dimension is the dimension to be collapsed
    Args:
        y: population output on a ring network. Numpy array (Batch, Units)
    Returns:
        Readout locations: Numpy array (Batch,)
    """
    pref = np.arange(0, 2*np.pi, 2*np.pi/y.shape[-1])  # preferences
    temp_sum = y.sum(axis=-1)
    temp_cos = np.sum(y*np.cos(pref), axis=-1)/temp_sum
    temp_sin = np.sum(y*np.sin(pref), axis=-1)/temp_sum
    loc = np.arctan2(temp_sin, temp_cos)
    return np.mod(loc, 2*np.pi)


def get_perf(all_y_hats, y_loc):
    """Get performance.
    Args:
      all_y_hats: Actual output. Numpy array (Batch, Time, Unit, Trials)
      y_loc: Target output location (-1 for fixation).
        Numpy array (Time, Batch) # original dims
    Returns:
      perf: Numpy array (Batch,)
    """
    if len(all_y_hats.shape) != 4:
        raise ValueError('pr450 edit: all_y_hats must have shape (Batch, Time, Unit, Trials)')

    assert len(y_loc.shape) == 2

    all_y_hats = all_y_hats.transpose(1, 0, 2, 3)
    y_loc = y_loc[-1]

    perfs_per_trial = []
    fixation_ignored_perfs_per_trial = []

    for trial_num in range(all_y_hats.shape[-1]):

        y_hat = all_y_hats[..., trial_num]

        # Only look at last time points
        y_hat = y_hat[-1]

        # Fixation and location of y_hat
        y_hat_fix = y_hat[..., 0]
        y_hat_loc = popvec(y_hat[..., 1:])

        # Fixating? Correctly saccading?
        fixating = y_hat_fix > 0.5

        original_dist = y_loc - y_hat_loc
        dist = np.minimum(abs(original_dist), 2*np.pi-abs(original_dist))
        corr_loc = dist < 0.2*np.pi

        # Should fixate?
        should_fix = y_loc < 0

        # performance
        perf = should_fix * fixating + (1-should_fix) * corr_loc * (1-fixating)

        perfs_per_trial.append(perf)
        fixation_ignored_perfs_per_trial.append(corr_loc)

    return np.mean(perfs_per_trial), np.mean(fixation_ignored_perfs_per_trial)

