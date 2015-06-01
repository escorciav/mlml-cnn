import os

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

def create_id(name, dirname, aux_dir=AUX_DIR):
    aux_dir = os.path.join(dirname, aux_dir)
    raise NotImplementedError, 'Work here'
    if not os.path.isdir(aux_dir):
        return 0

def dump_annotation_batches(name, Y, batch_size=256):
    """Save HDF5 files used for caffe-stochastic solver"""
    raise NotImplementedError, 'Work here'

def load_labels(dirname):
    """Load train/test label matrix"""
    filename = os.path.join(dirname, 'label_train.h5')
    train = utils.h5py_load(filename, 'labels')
    filename = os.path.join(dirname, 'label_test.h5')
    test = utils.h5py_load(filename, 'labels')
    return train, test

def main(exp_id='00', data_dir=DATA_DIR, flip_prob=FLIP_PROB,
        flip_type=FLIP_TYPE):
    check_txt_sources(data_dir)
    check_image_labels(data_dir)
    train_id, test_id = create_id(exp_id, data_dir)
    Y_train, Y_test = load_labels(data_dir)
    Y_train_f = flip_label_matrix(Y_train, flip_prob, flip_type)
    dump_annotation_batches(train_id, Y_train_f, train_batch_sz)
    dump_annotation_batches(test_id, Y_test, test_batch_sz)

if __name__ == '__main__':
    main()
