import numpy as np

def flip_labels(arr, thr, only_pos=False, rng=None):
    """ Flip instances of a label matrix randomly

    Parameters
    ----------
    arr : ndarray
        Label matrix with entries belong to {0, 1}
    thr : float
        Percentage of labels to flip
    only_pos : bool (optional, dflt=False)
        Flip only positivie (1) entries
    rng : np random class (optional, dflt=None)

    Returns
    -------
    f_arr : ndarray (float32)

    """
    if rng is None:
       rng = np.random.RandomState()

    arr_f = rng.uniform(size=arr.shape)
    if only_pos:
        # !!! Potential error !!!
        idx = np.logical_and(arr_f < thr, arr)
        arr_f[idx] = 0
    else:
        arr_f = np.logical_xor(arr_f < thr, arr)
    return np.float32(arr_f)
