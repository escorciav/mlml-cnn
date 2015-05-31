import os
import unittest

import pytest

import vgg16

def test_vgg16_multilabel_hdf5():
    filename = 'hola.txt'
    trans_dict = dict(crop_size=224, mean_value=[104, 117, 123], mirror=True)
    prm = {'batch_size':256, 'input':'image_data_no_label',
        'label_src':'target.txt', 'img_src':'img.txt', 'img_root':'data/',
        'img_transf':trans_dict, 'loss':'l2-norm'}
    # Return a NetSpec instance
    net = vgg16.vgg16_multilabel_hdf5(None, **prm)
    # Dump network spec into txt
    vgg16.vgg16_multilabel_hdf5(filename, **prm)
    if os.path.isfile(filename):
        os.remove(filename)

if __name__ == '__main__':
    unittest.main()

