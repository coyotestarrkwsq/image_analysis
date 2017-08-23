#!/usr/bin/env python

"""example usage: \
data_to_tf.py -i ~/data_set/mitos14/validation/ -o ~/data_set/mitos14/validation/val.record"""

import tensorflow as tf
import csv
import glob
from object_detection.utils import dataset_util
import argparse
import os
import scipy.misc as misc
import sys

def check_min_dim(coord, dim):
  if coord < 0:
    return float(0)
  else:
    return coord/dim

def check_max_dim(coord, dim):
  if coord > dim:
    return float(1)
  else:
    return coord/dim

def create_tf_example(example):

  filepath, filename = os.path.split(example)
  filtername, exts = os.path.splitext(filename)

  image = misc.imread(example)
  height = image.shape[0]# Image height
  width = image.shape[1] # Image width
  
  mitosis = filepath + '/' + filtername + '_mitosis.csv'
  with open(mitosis, 'rb') as csvfile:
    mitosis = csv.reader(csvfile)
    mitosis = list(mitosis)
  
  non_mitosis = filepath + '/' + filtername + '_not_mitosis.csv' 
  with open(non_mitosis, 'rb') as csvfile:
    non_mitosis = csv.reader(csvfile)
    non_mitosis = list(non_mitosis)

  with tf.gfile.GFile(example, 'rb') as fid:
    encoded_image_data = fid.read()
  image_format = b'jpg' 

  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = [] # List of string class name of bounding box (1 per box)
  classes = [] # List of integer class id of bounding box (1 per box)


  for coord in mitosis:
    xmin = (float(coord[0])-32)
    xmin = check_min_dim(xmin, width)

    ymin = (float(coord[1])-32)
    ymin = check_min_dim(ymin, height)

    xmax = (float(coord[0])+32)
    xmax = check_max_dim(xmax, width)

    ymax = (float(coord[1])+32)
    ymax = check_max_dim(ymax, height)


    xmins.append(xmin)
    xmaxs.append(xmax)

    ymaxs.append(ymax)
    ymins.append(ymin)

    classes_text.append('mitosis')
    classes.append(1)


  for coord in non_mitosis:
    xmin = (float(coord[0])-32)
    xmin = check_min_dim(xmin, width)

    ymin = (float(coord[1])-32)
    ymin = check_min_dim(ymin, height)

    xmax = (float(coord[0])+32)
    xmax = check_max_dim(xmax, width)

    ymax = (float(coord[1])+32)
    ymax = check_max_dim(ymax, height)


    xmins.append(xmin)
    xmaxs.append(xmax)

    ymaxs.append(ymax)
    ymins.append(ymin)

    classes_text.append('non_mitosis')
    classes.append(2)


  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example


def main(_):
  examples = glob.glob(FLAGS.image_path + '*jpg') 

  writer = tf.python_io.TFRecordWriter(FLAGS.save_path)

  for example in examples:
    tf_example = create_tf_example(example)
    writer.write(tf_example.SerializeToString())

  writer.close()

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description = __doc__)


  parser.add_argument(
        '-i',
        required = 'True',
        dest = 'image_path',
        help = 'path of image files'
    )

  parser.add_argument(
        '-o',
        required = 'True',
        dest = 'save_path',
        help = 'save path and file name'
    )

  FLAGS = parser.parse_args()
  main(sys.argv[1:])
