import tensorflow as tf
import os
import random

def data_reader(input_dir, shuffle=True):
  """Read images from input_dir then shuffle them
  Args:
    input_dir: string, path of input dir, e.g., /path/to/dir
  Returns:
    file_paths: list of strings
    names: list of strings
    labels: list of integers
  """
  file_paths = []
  file_names = []
  labels = [] # int
  names = [] # readable label

  label_index = 0
  category_count = 0

  for category in os.scandir(input_dir):
    if category.is_dir():
      category_count += 1
      for img_file in os.scandir(category.path):
        if img_file.name.endswith('.jpg') and img_file.is_file():
          file_paths.append(img_file.path)
          names.append(category.name)
          labels.append(label_index)
      label_index += 1

  if shuffle:
    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to label in the
    # saved TFRecord files. Make the randomization repeatable.
    shuffled_index = list(range(len(file_paths)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    file_paths = [file_paths[i] for i in shuffled_index]
    names = [names[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

  assert len(file_paths) == len(names)
  assert len(file_paths) == len(labels)

  return file_paths, names, labels


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(file_path, image_buffer, label, name):
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
      'image/class/label': _int64_feature(label),
      'image/class/name': _bytes_feature(tf.compat.as_bytes(name)),
      'image/file_name': _bytes_feature(tf.compat.as_bytes(os.path.basename(file_name))),
      'image/encoded_image': _bytes_feature((image_buffer))
    }))
  return example

def data_writer(input_dir):
  """Write data to tfrecords
  """
  file_paths, names, labels = data_reader(input_dir)

  tfrecords_dir = 'data/tfrecords'
  os.makedirs(tfrecords_dir, exist_ok=True)

  images_num = len(file_paths)
  train_num = (int)(images_num * 0.8) # use 80% images for training
  valid_num = 0

  train_writer = tf.python_io.TFRecordWriter(tfrecords_dir + 'train.tfrecords')
  valid_writer = tf.python_io.TFRecordWriter(tfrecords_dir + 'valid.tfrecords')


  for i in range(len(file_paths)):
    file_path = file_paths[i]
    label = labels[i]
    name = names[i]

    with tf.gfile.FastGFile(file_path, 'r') as f:
      image_data = f.read()

    example = _convert_to_example(file_path, image_data, label, name)

    if i < train_num:
      train_writer.write(example.SerializeToString())
    else:
      valid_num += 1
      valid_writer.write(example.SerializeToString())

    if i % 500 == 0:
      print("Processed {}/{}.".format(i, images_num))

  print("Train num: {}".format(train_num))
  print("Valid num: {}".format(valid_num))
  train_writer.close()
  valid_writer.close()

if __name__ == '__main__':
  input_dir = 'data/raw'
  data_writer(input_dir)
