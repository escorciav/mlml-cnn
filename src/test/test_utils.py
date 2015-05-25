import os
import unittest

import numpy as np
import pytest

import utils

def test_flip_labels():
    # Dummy test when only_pos is True 
    arr = np.array([[0, 0, 1], [1, 1, 0], [1, 0, 1], [1, 1, 1], [1, 0, 0],
                    [0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]])
    thr = 0.2
    utils.flip_labels(arr, thr, only_pos=True)
    # Check pctg of labels flipped when only_pos is False
    msg = 'Big difference of labels (un)flipped'
    for thr in [0.5, 0.75]:
        arr_f = utils.flip_labels(arr, thr)
        nflips = (arr != arr_f).sum()
        assert abs(nflips - arr.size*thr) <= arr.size*thr*0.5, msg 
    """ TODO: formal test is fucking randomly
    rng = np.random.RandomState(313)
    """

if __name__ == '__main__':
    unittest.main()

