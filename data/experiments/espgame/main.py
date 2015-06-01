import os

import numpy as np

import datasets as ds
import utils
from utils import flip_labels as flip_label_matrix

AUX_DIR = 'aux'
FLIP_PROB, FLIP_TYPE = 0.0, True
DATA_DIR = 'data/experiments/espgame/data'
TRAIN_LIST = 'data/ESP-Game/espgame_train_list.txt'
TEST_LIST = 'data/ESP-Game/espgame_test_list.txt'
TRAIN_ANNOT = 'data/ESP-Game/espgame_train_annot.hvecs'
TEST_ANNOT = 'data/ESP-Game/espgame_test_annot.hvecs'
DATASET_INFO = dict(train_list=TRAIN_LIST, test_list=TEST_LIST,
    train_annot=TRAIN_ANNOT, test_annot=TEST_ANNOT)

def check_image_labels(dirname, prm=DATASET_INFO):
    """Create/Check that ESP-Game labels are in HDF5 format

    Note
    ----
    Be careful launching multiple-process dumping source files

    """
    filename = os.path.join(dirname, 'label_train.h5')
    if not os.path.isfile(filename):
        ds.espgame_dump_labels(prm['train_annot'], filename)
    filename = os.path.join(dirname, 'label_test.h5')
    if not os.path.isfile(filename):
        ds.espgame_dump_labels(prm['test_annot'], filename)

def check_txt_sources(dirname, prm=DATASET_INFO):
    """Create/Check that if train and test list exist

    Note
    ----
    Be careful launching multiple-process dumping source files

    """
    filename = os.path.join(dirname, 'img_train.txt')
    if not os.path.isfile(filename):
        ds.espgame_dump_list(prm['train_list'], filename)
    filename = os.path.join(dirname, 'img_test.txt')
    if not os.path.isfile(filename):
        ds.espgame_dump_list(prm['test_list'], filename)

def create_prefix(name, dirname, aux_dir):
    """Create prefix to identify and experiment"""
    aux_dir = os.path.join(dirname, aux_dir)
    exp_id = os.path.join(aux_dir, name)
    if not os.path.isdir(aux_dir):
        os.makedirs(aux_dir)
    return exp_id

def dump_annotation_batches(name, Y, prefix=DATA_DIR, aux_dir=AUX_DIR,
        clobber=True, txt=True):
    """Save HDF5 files used for caffe-stochastic solver"""
    exp_id = create_prefix(name, prefix, aux_dir)
    src_file, h5_file = exp_id + '.txt', exp_id + '.h5'
    src_exist = os.path.isfile(src_file)
    if clobber or not src_exist:
        utils.h5py_save(h5_file, h5mode='w', label=np.float32(Y.T))
    if txt and not src_exist:
        with open(src_file, 'w') as fid:
            fid.write(src_file)
    return src_file

def load_labels(dirname):
    """Load train/test label matrix"""
    filename = os.path.join(dirname, 'label_train.h5')
    train = utils.h5py_load(filename, 'labels')
    filename = os.path.join(dirname, 'label_test.h5')
    test = utils.h5py_load(filename, 'labels')
    return train, test

def main(exp_id='00', data_dir=DATA_DIR, flip_prob=FLIP_PROB,
        flip_type=FLIP_TYPE):
    train_id, test_id = exp_id + '_trn', exp_id + '_tst'
    # Check if source and annotation files exist
    check_txt_sources(data_dir)
    check_image_labels(data_dir)
    # Load and Flip label matrix
    Y_train, Y_test = load_labels(data_dir)
    Yf_train = flip_label_matrix(Y_train, flip_prob, flip_type)
    # Dump annotations in caffe-hdf5 format
    h5_src_train = dump_annotation_batches(train_id, Yf_train, prefix=data_dir)
    h5_src_test = dump_annotation_batches(test_id, Y_test, prefix=data_dir)
    # Create/Update network prototxt
    # Create/Update solver prototxt
    # Launch process

if __name__ == '__main__':
    main()
