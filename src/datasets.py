import glob
import os

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

