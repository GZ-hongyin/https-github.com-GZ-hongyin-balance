import os
import glob
import numpy as np
from shutil import copyfile
symlink = True    # If this is false the files are copied instead

cinic_directory = "/home/cvk4_n1/douli/cinic-10"
enlarge_directory = "/home/cvk4_n1/douli/cinic-10-trainlarge"
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
sets = ['train', 'valid', 'test']
if not os.path.exists(enlarge_directory):
    os.makedirs(enlarge_directory)
if not os.path.exists(enlarge_directory + '/train'):
    os.makedirs(enlarge_directory + '/train')
if not os.path.exists(enlarge_directory + '/test'):
    os.makedirs(enlarge_directory + '/test')

for c in classes:
    if not os.path.exists('{}/train/{}'.format(enlarge_directory, c)):
        os.makedirs('{}/train/{}'.format(enlarge_directory, c))
    if not os.path.exists('{}/test/{}'.format(enlarge_directory, c)):
        os.makedirs('{}/test/{}'.format(enlarge_directory, c))

for s in sets:
    for c in classes:
        source_directory = '{}/{}/{}'.format(cinic_directory, s, c)
        filenames = glob.glob('{}/*.png'.format(source_directory))
        for fn in filenames:
            dest_fn = fn.split('/')[-1]
            if s == 'train' or s == 'valid':
                dest_fn = '{}/train/{}/{}'.format(enlarge_directory, c, dest_fn)
                if symlink:
                    if not os.path.islink(dest_fn):
                        os.symlink(fn, dest_fn)
                else:
                    copyfile(fn, dest_fn)


            elif s == 'test':
                dest_fn = '{}/test/{}/{}'.format(enlarge_directory, c, dest_fn)
                if symlink:
                    if not os.path.islink(dest_fn):
                        os.symlink(fn, dest_fn)
                else:
                    copyfile(fn, dest_fn)