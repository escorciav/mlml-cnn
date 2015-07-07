import argparse
import os
import subprocess

from main import EXP_DIR

PLT_SCRIPT = os.path.join('tools', 'plot_script.sh')

def input_parser():
    help_id = 'ID used to identify experiment and its results'
    help_pl = 'List of graphs to create'
    help_c = 'Remove temporal files'

    p = argparse.ArgumentParser(
       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('exp_id', help=help_id, type=str)
    p.add_argument('-pl', '--plt_list', help=help_pl, default=[6], nargs='+')
    p.add_argument('-c', '--clean', help=help_c, action='store_false')
    return p

def main(exp_id, plt_list, clean):
    exp_folder = os.path.join(EXP_DIR, exp_id)
    logfile = os.path.join(exp_folder, exp_id + '.log')
    for i in plt_list:
        figname = os.path.join(exp_folder, '{0}_{1}.png'.format(exp_id, i))
        cmd = [PLT_SCRIPT, i, figname, logfile]
        status = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        if clean:
            logprefix = exp_id + '.log'
            os.remove(logprefix + '.train') 
            os.remove(logprefix + '.test')

if __name__ == '__main__':
    p = input_parser()
    args = p.parse_args()
    main(**vars(args))
