import tensorflow as tf
import ops

class Discriminator:
  def __init__(self, name, use_sigmoid=False):
    self.name = name
    self.reuse = False
    self.use_sigmoid = use_sigmoid

  def __call__(self, input):
    """
    Args:
      input: batch_size x image_size x image_size x 3
    Returns:
      output: 4D tensor batch_size x image_size/16 x image_size/16 x 1
              filled with 0.9 if real, 0.0 if fake
    """
    with tf.variable_scope(self.name):
      C64 = ops.Ck(input, 64, reuse=self.reuse, use_batchnorm=False, name='C64') # (?, 64, 64, 64)
      C128 = ops.Ck(C64, 128, reuse=self.reuse, name='C128')                     # (?, 32, 32, 128)
      C256 = ops.Ck(C128, 256, reuse=self.reuse, name='C256')                    # (?, 16, 16, 256)
      C512 = ops.Ck(C256, 512, reuse=self.reuse, name='C512')                    # (?, 8, 8, 512)

      # apply a convolution to produce a 1 dimensional output (1 channel?)
      # set use_sigmoid = False if use_lsgan == True
      output = ops.last_conv(C512,
          reuse=self.reuse, use_sigmoid=self.use_sigmoid, name='output')         # (?, 8, 8, 1)

    self.reuse = True
    self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    return output
