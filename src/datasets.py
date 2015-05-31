import glob
import os

from oct2py import octave

import utils

def espgame_dump_list(filename, outfile, ext='.jpg', root=None):
    fid_o = open(outfile, 'w')
    with open(filename, 'r') as fid_i:
        for line in fid_i:
            line = line.strip('\n').split('/')[1]
            if root is not None:
                line = glob.glob(os.path.join(root, line + '*'))
                line = os.path.basename(line[0])
            else:
                line += ext
            fid_o.write(line + '\n')
    fid_o.close()

def espgame_dump_labels(filename, outfile, attrb='labels',
        dirname='data/ESP-Game/'):
    # Add ESP-game dataset to octave path
    octave.addpath(dirname)
    # Use vec_read octave-function to load data
    info = octave.vec_read(filename)
    # Dump data into a standard format
    utils.h5py_save(outfile, **{attrb:info.T})

