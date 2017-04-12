import tensorflow as tf

def convert2int(image):
  """ Transfrom from float tensor ([-1.,1.]) to int image ([0,255])
  """
  return tf.image.convert_image_dtype((image+1.0)/2.0, tf.uint8)

def convert2float(image):
  """ Transfrom from int image ([0,255]) to float tensor ([-1.,1.])
  """
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  return (image/127.5) - 1.0

def batch_convert2int(images):
  """
  Args:
    images: 4D float tensor (batch_size, image_size, image_size, depth)
  Returns:
    4D int tensor
  """
  return tf.map_fn(convert2int, images, dtype=tf.uint8)

def batch_convert2float(images):
  """
  Args:
    images: 4D int tensor (batch_size, image_size, image_size, depth)
  Returns:
    4D float tensor
  """
  return tf.map_fn(convert2float, images, dtype=tf.float32)
