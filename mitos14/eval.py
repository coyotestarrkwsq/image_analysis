#!/usr/bin/env python

"""example usage: \
eval.py -i /home/wangsq/data_set/mitos14/validation/ -o /home/wangsq/data_set/mitos14/result/ \
-m /home/wangsq/data_set/mitos14/testrun2/output_inference_graph.pb/frozen_inference_graph.pb \
-l /home/wangsq/data_set/mitos14/testrun/data/mitos14_label_map.pbtxt -c 2 -g 1"""


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
from sklearn.metrics import precision_recall_curve







def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def check_min_dim(coord):
  if coord < 0:
    return float(0)
  else:
    return coord

def check_max_dim(coord, dim):
  if coord > dim:
    return float(dim)
  else:
    return coord

def draw_text_on_image_array(image_np, image_pil, x, y, color):
  draw = ImageDraw.Draw(image_pil)
  width, height = image_pil.size

  try:
    font = ImageFont.truetype('arial.ttf', 24)
  except IOError:
    font = ImageFont.load_default()
  msg = str(x) + ',' + str(y)
  draw.text((x,y), msg, font = font, fill=color)


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

def groundtruth_boxes(width, height, x, y):
  xmin = (float(x)-32)
  xmin = check_min_dim(xmin)
 
  ymin = (float(y)-32)
  ymin = check_min_dim(ymin)

  xmax = (float(x)+32)
  xmax = check_max_dim(xmax, width)

  ymax = (float(y)+32)
  ymax = check_max_dim(ymax, height)

  return [ymin / height, xmin / width, ymax / height, xmax / width]

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

  evaluator = object_detection_evaluation.ObjectDetectionEvaluation(2)

  PATH_TO_CKPT = FLAGS.model_path
  PATH_TO_LABELS = FLAGS.label_path
  NUM_CLASSES = int(FLAGS.no_class)

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
        
        boxes = np.squeeze(boxes)
    
        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)
       
        
        if ground_truth:

          image_pil = Image.fromarray(np.uint8(image_np)).convert('RGB')
          width, height = image_pil.size

          mitosis = filepath + '/' + filtername + '_mitosis.csv'

          with open(mitosis, 'rb') as csvfile:
            mitosis = csv.reader(csvfile)
            mitosis = list(mitosis)
          if len(mitosis) > 0:  
            gt_boxes = np.array([[0.0, 0.0, 0.0, 0.0] for _ in range(len(mitosis))])
          else:
            gt_boxes = np.zeros((0,4))

          gt_labels = np.array([1 for _ in range(len(mitosis))])
          
          for i, coord in enumerate(mitosis):
            x = int(coord[0])
            y = int(coord[1])
            draw_text_on_image_array(image_np, image_pil, x, y, color = 'Yellow')
            gt_boxes[i] = groundtruth_boxes(width, height, x, y)

        

          evaluator.add_single_ground_truth_image_info(
          im_id, gt_boxes, gt_labels)

          det_class = []
          det_boxes = []
          det_scores = []

          for i, j in enumerate(classes):
          
            if j == 1:
              det_class.append(1)
              det_boxes.append(boxes[i].tolist())
              det_scores.append(scores[i])


          
          
          det_class = np.array(det_class)    

          #det_boxes = map(normalize, det_boxes)

          det_boxes = np.array(det_boxes)
          

          det_scores = np.array(det_scores)

          evaluator.add_single_detected_image_info(
            im_id, det_boxes,
            det_scores,
            det_class)

        # Visualization of the results of a detection.
        csvfile = save_path + filtername + '.csv'
        with open(csvfile, 'w') as f:
          visualize_boxes_and_labels_on_image_array(
              image_np,
              boxes,
              classes,
              scores,
              category_index,
              csvfile = f,
              use_normalized_coordinates=True,
              line_thickness=4)

        #plt.figure(figsize=IMAGE_SIZE)
        #plt.imshow(image_np)
        #plt.show()
        #sys.exit()

        misc.imsave(save_path + filename, image_np)

  if ground_truth:
    scores = np.concatenate(evaluator.scores_per_class[1])

    tp_fp_labels = np.concatenate(evaluator.tp_fp_labels_per_class[1])
    
    precision, recall, thresholds = precision_recall_curve(tp_fp_labels, scores)

    precision = precision[:-1]
    
    recall = recall[:-1]

    f1 = 2 * ((precision * recall) / (precision + recall))
    import matplotlib.pyplot as plt

    plt.plot(thresholds, precision, 'r')
    plt.plot(thresholds, recall, 'b')
    plt.plot(thresholds, f1, 'g')
    #plt.axvline(0.5)


    plt.xlabel('threshold')
    plt.ylabel('recall / precision / f1 score')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title('precision : red, recall : blue, f1-score : green')
    plt.savefig(save_path + 'threhold_vs_pr.png')

    plt.clf()  



    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.savefig(save_path + 'precision_recall.png')

    

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
        '-m',
        required = 'True',
        dest = 'model_path',
        help = 'graph model path'
    )

  parser.add_argument(
        '-c',
        required = 'True',
        dest = 'no_class',
        help = 'number of classes'
    )

  parser.add_argument(
        '-l',
        required = 'True',
        dest = 'label_path',
        help = 'label path'
    )

  parser.add_argument(
        '-g',
        required = 'True',
        dest = 'g_truth',
        help = 'indicate if want to show ground truth, 1 or 0'
    )

  FLAGS = parser.parse_args()
  main(sys.argv[1:])
