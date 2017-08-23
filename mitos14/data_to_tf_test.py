import os

import numpy as np
import PIL.Image
import tensorflow as tf
import glob
from data_to_tf import create_tf_example

class DataToTfTest(tf.test.TestCase):

  def _assertProtoEqual(self, proto_field, expectation):
    """Helper function to assert if a proto field equals some value.

    Args:
      proto_field: The protobuf field to compare.
      expectation: The expected value of the protobuf field.
    """
    proto_list = [p for p in proto_field]
    self.assertListEqual(proto_list, expectation)

  def test_data_to_tf_example(self):


   
    height = 1376
    width = 1539
    image_file_name = 'A03_00Aa.jpg'
    
    xmin1 = (float(1094)-32)/width
    xmax1 = (float(1094)+32)/width
    ymin1 = (float(1223)-32)/height
    ymax1 = (float(1223)+32)/height

    xmin2 = (float(1375)-32)/width
    xmax2 = (float(1375)+32)/width
    ymin2 = (float(542)-32)/height
    ymax2 = (float(542)+32)/height

    examples = '/home/wangsq/image_analysis/mitos14/test/A03_00Aa.jpg'

    example = create_tf_example(examples)

    self._assertProtoEqual(
        example.features.feature['image/height'].int64_list.value, [height])  
    self._assertProtoEqual(
        example.features.feature['image/width'].int64_list.value, [width])
    self._assertProtoEqual(
        example.features.feature['image/filename'].bytes_list.value,
        [image_file_name])
    self._assertProtoEqual(
        example.features.feature['image/source_id'].bytes_list.value,
        [image_file_name])  
    self._assertProtoEqual(
        example.features.feature['image/format'].bytes_list.value, ['jpg'])
    
    self._assertProtoEqual(
        example.features.feature['image/object/bbox/xmin'].float_list.value,
        [xmin1, xmin2])
    
    self._assertProtoEqual(
        example.features.feature['image/object/bbox/ymin'].float_list.value,
        [ymin1,ymin2])
    self._assertProtoEqual(
    
        example.features.feature['image/object/bbox/xmax'].float_list.value,
        [xmax1, xmax2])
    
    self._assertProtoEqual(
        example.features.feature['image/object/bbox/ymax'].float_list.value,
        [ymax1, ymax2])
    
    self._assertProtoEqual(
        example.features.feature['image/object/class/text'].bytes_list.value,
        ['mitosis', 'non_mitosis'])
    
    self._assertProtoEqual(
        example.features.feature['image/object/class/label'].int64_list.value,
        [1, 2])
    
  
    
   


if __name__ == '__main__':
  tf.test.main()
