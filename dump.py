import tensorflow as tf
import os
import random

def data_reader(input_dir, shuffle=True):
  """Read images from input_dir then shuffle them
  Args:
    input_dir: string, path of input dir, e.g., /path/to/dir
  Returns:
    file_paths: list of strings
  """
  file_paths = []

  for img_file in os.scandir(input_dir):
    if img_file.name.endswith('.jpg') and img_file.is_file():
      file_paths.append(img_file.path)

  if shuffle:
    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to label in the
    # saved TFRecord files. Make the randomization repeatable.
    shuffled_index = list(range(len(file_paths)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    file_paths = [file_paths[i] for i in shuffled_index]

  return file_paths


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(file_path, image_buffer):
  """Build an Example proto for an example.
  Args:
    file_path: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    label: integer, identifier for the ground truth for the network
    name: string, unique human-readable, e.g. 'Orange'
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """
  file_name = file_path.split('/')[-1]

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/file_name': _bytes_feature(tf.compat.as_bytes(os.path.basename(file_name))),
      'image/encoded_image': _bytes_feature((image_buffer))
    }))
  return example

def data_writer(input_dir):
  """Write data to tfrecords
  """
  file_paths = data_reader(input_dir)
  file_name = input_dir.split('/')[-1]

  tfrecords_dir = 'data/tfrecords/'
  os.makedirs(tfrecords_dir, exist_ok=True)

  images_num = len(file_paths)

  writer = tf.python_io.TFRecordWriter(tfrecords_dir + '{}.tfrecords'.format(file_name))

  for i in range(len(file_paths)):
    file_path = file_paths[i]

    with tf.gfile.FastGFile(file_path, 'r') as f:
      image_data = f.read()

    example = _convert_to_example(file_path, image_data)
    writer.write(example.SerializeToString())

    if i % 500 == 0:
      print("Processed {}/{}.".format(i, images_num))

  writer.close()

if __name__ == '__main__':
  print("Dump apple data...")
  input_dir = '/tmp/data/apple'
  data_writer(input_dir)
  print("Dump orange data...")
  input_dir = '/tmp/data/orange'
  data_writer(input_dir)
