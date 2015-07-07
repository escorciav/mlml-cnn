import argparse
import os
import subprocess

import numpy as np
from jinja2 import Template

import datasets as ds
import utils
from vgg16 import vgg16_multilabel_hdf5
from utils import flip_labels as flip_label_matrix

EXP_DIR = os.path.join('data', 'experiments', 'espgame')
DS_DIR = os.path.join('data', 'ESP-Game')
AUX_DIR = os.path.join(EXP_DIR, 'aux')
PROTOTXT_NET = os.path.join(AUX_DIR, 'vgg16_multilabel_00.jinja2')
PROTOTXT_SOLVER = os.path.join(AUX_DIR, 'vgg16_solver_00.jinja2')
SNAPSHOT_FILE = os.path.join(EXP_DIR, '..', 'models',
    'VGG_ILSVRC_16_layers.caffemodel')

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

def create_prototxt_net(filename, version=0, **kwargs):
    if version == 0:
        prm = dict(batch_size=64, label_src="{{ h5_src_train }}",
            input='image_data_no_label', img_src="{{ img_src_train }}",
            img_root='data/ESP-Game/ESP-ImageSet/', loss='l2-norm',
            img_transf=dict(crop_size=224, mean_value=[104, 117, 123],
            mirror=True), new_width=256, new_height=256, n_output=268)
        vgg16_multilabel_hdf5(filename, **prm)

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

def launch_caffe(name, solver, net, snapshot=None, finetune=False, gpu_id=-1,
        prefix=''):
    """Laucn caffe binary (finetunning)"""
    log = os.path.join(prefix, name + '.log')
    cmd = ['sh', 'train.sh', str(gpu_id), solver, net, log]
    if snapshot is not None:
       cmd += [snapshot]
       if finetune:
           cmd += ['1']
    status = subprocess.check_output(cmd, stderr=subprocess.STDOUT)

def load_labels(dirname):
    """Load train/test label matrix"""
    filename = os.path.join(dirname, 'label_train.h5')
    train = utils.h5py_load(filename, 'labels')
    filename = os.path.join(dirname, 'label_test.h5')
    test = utils.h5py_load(filename, 'labels')
    return train, test

def update_net_prototxt(txt_template, name, prefix, h5_train, h5_test, img_train,
        img_test):
    """Update network prototxt template"""
    with open(txt_template, 'r') as fid:
        prototxt = fid.read()
    template = Template(prototxt)
    netfile = os.path.join(prefix, name + '_net.prototxt')
    with open(netfile, 'w') as fid:
        print >>fid, template.render(img_src_train=img_train,
            img_src_test=img_test, h5_src_train=h5_train, h5_src_test=h5_test)
    return netfile

def update_solver_prototxt(txt_template, name, prefix, netfile):
    """Update solver prototxt template"""
    with open(txt_template, 'r') as fid:
        prototxt = fid.read()
    template = Template(prototxt)
    solverfile = os.path.join(prefix, name + '_solver.prototxt')
    snapshot = os.path.join(prefix, name + '_')
    with open(solverfile, 'w') as fid:
        print >>fid, template.render(snapshot=snapshot, net_src=netfile)
    return solverfile

# Program

def input_parser():
    help_id = 'ID used to identify experiment and its results'
    help_gpu = 'Device ID of the GPU used for the experiment'
    help_fp = 'Probability of flipping labels'
    help_ft = 'Flipping strategy (True: just positive, False: All)'
    help_ff = 'Finetune model otherwise Resume training'
    help_pn = 'Fullpath of prototxt network jinja2 template'
    help_ps = 'Fullpath of prototxt solver jinja2 template'
    help_sm = 'Fullpath of snapshot caffe-model'
    help_ad = 'Fullpath of auxiliar folder of ESP-Game experiments'

    p = argparse.ArgumentParser(
       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('exp_id', help=help_id, type=str)
    p.add_argument('-gpu', '--gpu_id', help=help_gpu, type=int, default=0)
    p.add_argument('-fp', '--flip_prob', help=help_fp, type=float, default=0.0)
    p.add_argument('-ft', '--flip_type', help=help_ft, action='store_false')
    p.add_argument('-ff', '--finetune_flag', help=help_ff, action='store_false')
    p.add_argument('-pn', '--prototxt_net', help=help_pn,
        default=PROTOTXT_NET)
    p.add_argument('-ps', '--prototxt_solver', help=help_ps,
        default=PROTOTXT_SOLVER)
    p.add_argument('-sm', '--snapshot_file', help=help_sm,
        default=SNAPSHOT_FILE)
    p.add_argument('-ad', '--aux_dir', help=help_ad, default=AUX_DIR)
    return p

def main(exp_id, gpu_id, flip_prob, flip_type, finetune_flag,
        prototxt_net, prototxt_solver, aux_dir, snapshot_file):
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
    solverfile = update_solver_prototxt(prototxt_solver, exp_id, exp_dir,
        netfile)
    # Launch process
    launch_caffe(exp_id, solverfile, netfile, finetune=finetune_flag,
        gpu_id=gpu_id, prefix=exp_dir, snapshot=snapshot_file)

if __name__ == '__main__':
    p = input_parser()
    args = p.parse_args()
    main(**vars(args))
