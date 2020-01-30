from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf



IMAGE_SIZE = 200
IMAGE_SIZE_label = 51
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1000
HEIGHT=IMAGE_SIZE 
WIDTH=IMAGE_SIZE 
HEIGHT_label=IMAGE_SIZE_label 
WIDTH_label=IMAGE_SIZE_label 


NUMEVAL_SAMPLES=NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
height=HEIGHT
width=WIDTH

def read_fiber10(filename_queue):

  class CIFAR10Record(object):
    pass
  result = CIFAR10Record()

  result.height =HEIGHT 
  result.width = WIDTH
  result.height_label =HEIGHT_label 
  result.width_label = WIDTH_label
  result.depth = 2
  result.depth_label = 2
  image_bytes = result.height * result.width * result.depth 
  label_bytes = result.height_label * result.width_label * result.depth_label

  record_bytes = label_bytes + image_bytes


  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, value = reader.read(filename_queue)

  # Convert from a string to a vector of uint8 that is record_bytes long.
  record_bytes_label = tf.decode_raw(value, tf.uint8)
  record_bytes_image = tf.decode_raw(value, tf.uint8)
  depth_label_major = (tf.reshape(
      tf.slice(record_bytes_label, [0],
                       [label_bytes]),
      [result.depth,result.height_label, result.width_label]))
    
  depth_label_major=tf.transpose(depth_label_major,[1,2,0])
          
  
  depth_image_major = (tf.reshape(
      tf.slice(record_bytes_image, [label_bytes],
                       [image_bytes]),
      [result.depth,result.height, result.width]))
      
  depth_image_major=tf.transpose(depth_image_major,[1,2,0])
      
  result.uint8label = depth_label_major
  result.uint8image=depth_image_major


  return result


def _generate_image_and_label_batch(image0, label0, min_queue_examples,
                                    batch_size, shuffle):
  
  num_preprocess_threads = 16
  num_preprocess_threads_eval = 1
  if shuffle:
    images, labels = tf.train.shuffle_batch(
        [image0, label0],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, labels = tf.train.batch(
        [image0, label0],
        batch_size=batch_size,
        num_threads=num_preprocess_threads_eval,
        capacity=min_queue_examples + 3 * batch_size)



  return images, labels



def distorted_inputs(data_dir, batch_size,NUMTRAIN_SAMPLES):

  
  filenames = [os.path.join(data_dir, 'fiber_train_data_%d.bin' % i)
               for i in xrange(3, 4)]

  print(filenames)
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)


  filename_queue = tf.train.string_input_producer(filenames)

  with tf.name_scope('data_augmentation'):

    read_input = read_fiber10(filename_queue)

    reshaped_image = tf.image.convert_image_dtype(read_input.uint8image, tf.float32,saturate=False)
    reshaped_label = tf.image.convert_image_dtype(read_input.uint8label, tf.float32,saturate=False)




    
    reshaped_image.set_shape([HEIGHT, WIDTH, read_input.depth])
    reshaped_label.set_shape([HEIGHT_label, WIDTH_label, read_input.depth])

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUMTRAIN_SAMPLES *
                             min_fraction_of_examples_in_queue)
    print ('Filling queue with %d Fiber images before starting to train. '
           'This will take a few minutes.' % min_queue_examples)


  return _generate_image_and_label_batch(reshaped_image, reshaped_label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)


def inputs(eval_data, data_dir, batch_size):

  if not eval_data:
#    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
#                 for i in xrange(1, 2)]
    filenames = [os.path.join(data_dir, 'fiber_train_data_%d.bin' % i)
               for i in xrange(5, 6)]
    num_examples_per_epoch = NUMTRAIN_SAMPLES
  else:
    filenames = [os.path.join(data_dir, 'fiber_test_data_3.bin')]
    num_examples_per_epoch = NUMEVAL_SAMPLES

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  with tf.name_scope('input'):

    filename_queue = tf.train.string_input_producer(filenames,shuffle=False)


    read_input = read_fiber10(filename_queue)
    reshaped_image = tf.image.convert_image_dtype(read_input.uint8image, tf.float32,saturate=False)
    reshaped_label = tf.image.convert_image_dtype(read_input.uint8label, tf.float32,saturate=False)
    reshaped_image.set_shape([HEIGHT, WIDTH, read_input.depth])
    reshaped_label.set_shape([HEIGHT_label, WIDTH_label, read_input.depth])
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)


  return _generate_image_and_label_batch(reshaped_image, reshaped_label,
                                         min_queue_examples, batch_size,
                                         shuffle=False)
