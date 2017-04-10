import tensorflow as tf

class Reader():
  def __init__(self, tfrecords_file, image_size=128, min_queue_examples=1000, batch_size=32, num_threads=8):
    """
    Args:
      tfrecords_file: string, tfrecords file path
      min_queue_examples: integer, minimum number of samples to retain in the queue that provides of batches of examples
      batch_size: integer, number of images per batch
      num_threads: integer, number of preprocess threads
    """
    self.tfrecords_file = tfrecords_file
    self.image_size = image_size
    self.min_queue_examples = min_queue_examples
    self.batch_size = batch_size
    self.num_threads = num_threads
    self.reader = tf.TFRecordReader()

  def feed(self):
    """
    Returns: (images, labels, names)
      images: 4D tensor [batch_size, image_width, image_height, image_depth]
      labels: integer list with size=batch_size, e.g. [3,2,50,1]
      names: byte-string list with size=batch_size, e.g. [b'Orange', b'Apple']
    """
    with tf.name_scope('input'):
      filename_queue = tf.train.string_input_producer([self.tfrecords_file])
      reader = tf.TFRecordReader()

      _, serialized_example = self.reader.read(filename_queue)
      features = tf.parse_single_example(
          serialized_example,
          features={
            'image/class/label': tf.FixedLenFeature([], tf.int64),
            'image/class/name': tf.FixedLenFeature([], tf.string),
            'image/file_name': tf.FixedLenFeature([], tf.string),
            'image/encoded_image': tf.FixedLenFeature([], tf.string),
          })

      image_buffer = features['image/encoded_image']
      image = tf.image.decode_jpeg(image_buffer, channels=3)
      image = self._preprocess(image)
      label = features['image/class/label']
      name = features['image/class/name']
      images, labels, names = tf.train.shuffle_batch(
            [image, label, name], batch_size=self.batch_size, num_threads=self.num_threads,
            capacity=self.min_queue_examples + 3*self.batch_size,
            min_after_dequeue=self.min_queue_examples
          )

      tf.summary.image('images', images)
      norm_images = tf.subtract(tf.div(tf.image.resize_images(
          images, [s_size * 2 ** 4, s_size * 2 ** 4]), 127.5), 1.0)
    return norm_images, labels, names

  def _preprocess(self, image):
    image = tf.image.resize_image_with_crop_or_pad(image, size=(self.image_size, self.image_size))
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # image = tf.random_crop(image, [self.image_size, self.image_size, 3])
    # image = tf.image.random_brightness(image, max_delta=0.4)
    # image = tf.image.random_contrast(image, lower=0.6, upper=1.4)
    # image = tf.image.random_hue(image, max_delta=0.04)
    # image = tf.image.random_saturation(image, lower=0.6, upper=1.4)
    # image = tf.image.per_image_standardization(image)
    image.set_shape([self.image_size, self.image_size, 3])
    return image
