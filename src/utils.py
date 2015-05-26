import h5py
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

def h5py_load(filename, attribute):
    """Returns an attribute from an h5file
    """
    with h5py.File(filename, 'r') as h5file:
        return h5file[attribute][()]

def h5py_save(filename, h5mode='a',
              h5opt={'compression':'gzip', 'compression_opts':1}, **kwargs):
    """Save a dict as an h5file
    """
    with h5py.File(filename, h5mode) as h5file:
        for i in kwargs:
            h5file.create_dataset(i, data=kwargs[i], **h5opt)

def h5py_save_dict(filename, v, attribute='prm', mode='a'):
    """Serialize dict as json to save on hdf5 file
    """
    with h5py.File(filename, mode) as h5file:
        h5file[attribute] = json.dumps(v, sort_keys=True)
        h5file.attrs[attribute]='dict serialized as json'
