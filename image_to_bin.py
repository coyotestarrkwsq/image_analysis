#!/usr/bin/env python

from scipy import misc
import numpy as np
import argparse
import sys
import glob
import tensorflow as tf
import matplotlib.pyplot as plt

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))


def image_pad(image_arr, pad):
    return np.pad(image_arr, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)

def tfrecords_writer(filename, pad, image_addr, seg_addr):
    
    writer = tf.python_io.TFRecordWriter(filename)
    patch_dim = 2*pad+1

    for image, seg in zip(image_addr, seg_addr):
        image_arr = plt.imread(image)
        image_arr = image_pad(image_arr, pad)

        seg_arr = plt.imread(seg)
        
        for h in range(605):
            for w in range(700):
                new_image_arr = np.empty([patch_dim, patch_dim, 3])

                for c in range(3):
                    for i in range(patch_dim):
                        for j in range(patch_dim):
                            new_image_arr[i,j,c] = image_arr[h+i, w+j, c]
                
               
                #misc.toimage(new_image_arr).show()
                #sys.exit()
      
                label = seg_arr[h,w]
                feature = {'train/label': _int64_feature(label),
                        'train/image': _bytes_feature(tf.compat.as_bytes(new_image_arr.tostring()))}

                example = tf.train.Example(features = tf.train.Features(feature=feature))
                
                writer.write(example.SerializeToString())

    writer.close()

 
def main(_):
    
    train_filename = 'train.tfrecords'
    test_filename = 'test.tfrecords'

    image_addr = glob.glob(FLAGS.image_path)
    seg_addr = glob.glob(FLAGS.seg_path)
    pad = FLAGS.padding
    data_size = len(image_addr)
    train_end_index = int(FLAGS.training * data_size)    
    
    train_image_addr = image_addr[0:train_end_index] 
    train_seg_addr = seg_addr[0:train_end_index]
  
    test_start_index = train_end_index
 
    if FLAGS.validation != 0:
        test_start_index = int((FLAGS.training + FLAGS.validation) * data_size)
        
        val_image_addr = image_addr[train_end_index:test_start_index] 
        val_seg_addr = seg_addr[train_end_index:test_start_index]
        val_filename = 'val_tfrecords'
        
        tfrecords_writer(val_filename, pad, val_image_addr, val_seg_addr)

    test_image_addr = image_addr[test_start_index:] 
    test_seg_addr = seg_addr[test_start_index:]

    tfrecords_writer(train_filename, pad, train_image_addr, train_seg_addr)
    tfrecords_writer(test_filename, pad, test_image_addr, test_seg_addr)

    
    



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--image_path',
        required = 'True',
        help = 'path of image files, please enclose path with semicolon'
    )

    parser.add_argument(
        '--seg_path',
        required = 'True',
        help = 'path of seg image files, please enclose path with semicolon'
    )

    parser.add_argument(
        '--padding',
        type = int,
        default = 12,
        help = 'padding size'
    )

    parser.add_argument(
        '--validation',
        type = int,
        default = 0,
        help = 'validation set size (percent)'
    )

    parser.add_argument(
        '--training',
        type = float,
        default = 0.6,
        help = 'training set size (percent)'
    ) 


    FLAGS = parser.parse_args()
    main(sys.argv[1:])



