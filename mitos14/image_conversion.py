#!/usr/bin/env python

""" example usage: ./tif_to_jpg.py -i ~/Downloads/TUPAC16/data/01/test.tiff \
-s ~/Downloads/TUPAC16/data/01/jpg/ """

import scipy.misc as misc
import argparse
import sys
import os

def main(_):
    
    save_path = FLAGS.save_path
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    filepath, filename = os.path.split(FLAGS.image)
    filtername, exts = os.path.splitext(filename)    
    
    arr = misc.imread(FLAGS.image)
    misc.imsave(save_path + filtername + '.png', arr)



if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
            '-i',
            required = 'True',
            dest = 'image',
            help = 'image files'
    )

    parser.add_argument(
            '-s',
            required = 'True',
            dest = 'save_path',
            help = 'savepath'
    )

    FLAGS = parser.parse_args()
    main(sys.argv[1:])
