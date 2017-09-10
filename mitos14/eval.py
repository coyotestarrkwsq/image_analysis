#!/usr/bin/env python

"""example usage: \
data_to_tf.py -i /home/wangsq/data_set/mitos14/validation/ -o /home/wangsq/data_set/mitos14/result/ -g 1"""


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import csv
import math
import argparse

from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import scipy.misc as misc
import glob

from object_detection.utils import label_map_util
from object_detection.utils import object_detection_evaluation



PATH_TO_CKPT = '/home/wangsq/data_set/mitos14/testrun2/output_inference_graph.pb/frozen_inference_graph.pb'
PATH_TO_LABELS = '/home/wangsq/data_set/mitos14/testrun/data/mitos14_label_map.pbtxt'
NUM_CLASSES = 2

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')



label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def check_min_dim(coord):
  if coord < 0:
    return 0
  else:
    return coord

def check_max_dim(coord, dim):
  if coord > dim:
    return dim
  else:
    return coord

def draw_text_on_image_array(image_np, x, y, color):
  image_pil = Image.fromarray(np.uint8(image_np)).convert('RGB')
  draw = ImageDraw.Draw(image_pil)
  
  try:
    font = ImageFont.truetype('arial.ttf', 24)
  except IOError:
    font = ImageFont.load_default()
  msg = str(x) + ',' + str(y)
  draw.text((x,y), msg, font = font, fill=color)

  width, height = image_pil.size

  xmin = x-32
  xmin = check_min_dim(xmin)

  ymin = y-32
  ymin = check_min_dim(ymin)

  xmax = x+32
  xmax = check_max_dim(xmax, width)

  ymax = y+32
  ymax = check_max_dim(ymax, height)
  draw.line([(xmin, ymin), (xmin, ymax), (xmax, ymax),
             (xmax, ymin), (xmin, ymin)], width=8, fill=color)

  np.copyto(image_np, np.array(image_pil))


def visualize_boxes_and_labels_on_image_array(image,
                                              boxes,
                                              classes,
                                              scores,
                                              category_index,
                                              csvfile,
                                              use_normalized_coordinates=False,
                                              min_score_thresh=-1,
                                              line_thickness=2):

  mitosis = 'mitosis'
  for i in range(boxes.shape[0]):
    class_name = category_index[classes[i]]['name']
    if class_name == mitosis:
      if scores[i] > min_score_thresh:
        box = boxes[i].tolist()
        display_str = '{}: {}%'.format(
                class_name,
                int(100*scores[i]))
      

        ymin, xmin, ymax, xmax = box
        score = scores[i]

        draw_bounding_box_on_image_array(
          image,
          ymin,
          xmin,
          ymax,
          xmax,
          display_str,
          csvfile,
          score, 
          color='Green',
          thickness=line_thickness,
          use_normalized_coordinates=use_normalized_coordinates)



def draw_bounding_box_on_image_array(image,
                                     ymin,
                                     xmin,
                                     ymax,
                                     xmax,
                                     display_str,
                                     csvfile,
                                     score,
                                     color='red',
                                     thickness=2,
                                     use_normalized_coordinates=True):
  
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, display_str,
                             csvfile,
                             score,
                             color,
                             thickness, 
                             use_normalized_coordinates)
  np.copyto(image, np.array(image_pil))


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               display_str,
                               csvfile,
                               score,
                               color='red',
                               thickness=2,
                               use_normalized_coordinates=True):

  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  if use_normalized_coordinates:
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
  else:
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
  draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)
  try:
    font = ImageFont.truetype('arial.ttf', 24)
  except IOError:
    font = ImageFont.load_default()

  text_bottom = top
  text_width, text_height = font.getsize(display_str)
  margin = np.ceil(0.05 * text_height)
  draw.rectangle(
        [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                          text_bottom)],
        fill=color)
  draw.text(
        (left + margin, text_bottom - text_height - margin),
        display_str,
        fill='black',
        font=font)

  x = math.floor((right-left)/2)
  y = math.floor((bottom-top)/2)

  x = int(round(left) + x)
  y = int(round(top) + y)
  msg = str(x) + ',' + str(y)
  draw.text((x, y), msg, font=font,fill=color)

  csvfile.write(','.join(map(str, (x, y, score))) + '\n')



#PATH_TO_TEST_IMAGES_DIR = 'test_images'
#TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]



def main(_):
  TEST_IMAGE_PATHS = glob.glob(FLAGS.image_path + '*jpg')

  save_path = FLAGS.save_path

  ground_truth =int(FLAGS.g_truth)

  with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
      for im_id, image_path in enumerate(TEST_IMAGE_PATHS):
        image = Image.open(image_path)
        
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
        
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        
        #image_tensor is a placeholder tensor
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        
        # Each box represents a part of the image where a particular object was detected.
        # np arr of box coords with dim (1, num of detections, 4)
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        # np arr of scores of dim (1, num of detections)
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
              
        # np arr of classes of dim (1, num of detections)
        classes = detection_graph.get_tensor_by_name('detection_classes:0')

        # np arr of num_detections of dim 1
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        
        # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        
        filepath, filename = os.path.split(image_path)
        filtername, exts = os.path.splitext(filename)
        
        
       
        
        if ground_truth:
          mitosis = filepath + '/' + filtername + '_mitosis.csv'

          with open(mitosis, 'rb') as csvfile:
            mitosis = csv.reader(csvfile)
            mitosis = list(mitosis)

          for coord in mitosis:
            x = int(coord[0])
            y = int(coord[1])
            draw_text_on_image_array(image_np, x, y, color = 'Yellow')

        # Visualization of the results of a detection.
        csvfile = save_path + filtername + '.csv'
        with open(csvfile, 'w') as f:
          visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              csvfile = f,
              use_normalized_coordinates=True,
              line_thickness=4)

        #plt.figure(figsize=IMAGE_SIZE)
        #plt.imshow(image_np)
        #plt.show()
        #sys.exit()

        misc.imsave(save_path + filename, image_np)




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
        help = 'save path'
    )

  parser.add_argument(
        '-g',
        required = 'True',
        dest = 'g_truth',
        help = 'indicate if want to show ground truth, 1 or 0'
    )

  FLAGS = parser.parse_args()
  main(sys.argv[1:])
