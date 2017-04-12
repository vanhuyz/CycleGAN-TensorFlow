import tensorflow as tf

def convert2int(image):
  """ Transfrom from float image ([0.,1.]) to int image ([0,255])
  """
  return tf.image.convert_image_dtype((image+1.0)/2.0, tf.uint8)