import os

import numpy as np
from jinja2 import Template

import datasets as ds
import utils
from utils import flip_labels as flip_label_matrix

FLIP_PROB, FLIP_TYPE = 0.0, True
EXP_DIR = os.path.join('data', 'experiments', 'espgame')
DS_DIR = os.path.join('data', 'ESP-Game')
AUX_DIR = os.path.join(EXP_DIR, 'aux')
PROTOTXT_NET = os.path.join(AUX_DIR, 'vgg16_multilabel_00.jinja2')
TRAIN_LIST = os.path.join(DS_DIR, 'espgame_train_list.txt')
TEST_LIST = os.path.join(DS_DIR, 'espgame_test_list.txt')
TRAIN_ANNOT = os.path.join(DS_DIR, 'espgame_train_annot.hvecs')
TEST_ANNOT = os.path.join(DS_DIR, 'espgame_test_annot.hvecs')
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
    filename_train = os.path.join(dirname, 'img_train.txt')
    if not os.path.isfile(filename_train):
        ds.espgame_dump_list(prm['train_list'], filename_train)
    filename_test = os.path.join(dirname, 'img_test.txt')
    if not os.path.isfile(filename_test):
        ds.espgame_dump_list(prm['test_list'], filename_test)
    return filename_train, filename_test

def create_prefix(name, dirname):
    """Create prefix to identify and experiment"""
    exp_id = os.path.join(dirname, name)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    return exp_id

def dump_annotation_batches(name, Y, prefix=AUX_DIR, clobber=True, txt=True):
    """Save HDF5 files used for caffe-stochastic solver"""
    exp_id = create_prefix(name, prefix)
    src_file, h5_file = exp_id + '.txt', exp_id + '.h5'
    src_exist = os.path.isfile(src_file)
    if clobber or not src_exist:
        utils.h5py_save(h5_file, h5mode='w', label=np.float32(Y.T))
    if txt and not src_exist:
        with open(src_file, 'w') as fid:
            fid.write(h5_file)
    return src_file

def load_labels(dirname):
    """Load train/test label matrix"""
    filename = os.path.join(dirname, 'label_train.h5')
    train = utils.h5py_load(filename, 'labels')
    filename = os.path.join(dirname, 'label_test.h5')
    test = utils.h5py_load(filename, 'labels')
    return train, test

def update_net_prototxt(txt_template, name, prefix, h5_train, h5_test, img_train,
        img_test):
    """Update prototxt template"""
    with open(txt_template, 'r') as fid:
        prototxt = fid.read()
    template = Template(prototxt)
    netfile = os.path.join(prefix, name + '_net.prototxt')
    with open(netfile, 'w') as fid:
        print >>fid, template.render(img_src_train=img_train,
            img_src_test=img_test, h5_src_train=h5_train, h5_src_test=h5_test)
    return netfile

def main(exp_id='00', prototxt_net=PROTOTXT_NET, aux_dir=AUX_DIR,
        flip_prob=FLIP_PROB, flip_type=FLIP_TYPE):
    train_id, test_id = exp_id + '_trn', exp_id + '_tst'
    exp_dir = os.path.join(aux_dir, '..', exp_id)
    # Check if source and annotation files exist
    img_src_train, img_src_test = check_txt_sources(aux_dir)
    check_image_labels(aux_dir)
    # Load and Flip label matrix
    Y_train, Y_test = load_labels(aux_dir)
    Yf_train = flip_label_matrix(Y_train, flip_prob, flip_type)
    # Dump annotations in caffe-hdf5 format
    h5_src_train = dump_annotation_batches(train_id, Yf_train,
        prefix=exp_dir)
    h5_src_test = dump_annotation_batches(test_id, Y_test,
        prefix=exp_dir)
    # Update network prototxt
    netfile = update_net_prototxt(prototxt_net, exp_id, exp_dir,
        h5_train=h5_src_train, h5_test=h5_src_test, img_train=img_src_train,
        img_test=img_src_test)
    # Update solver prototxt
    # Launch process

if __name__ == '__main__':
    main()
