import tensorflow as tf
import ops

class Discriminator:
  def __init__(self, name):
    self.name = name
    self.reuse = False

  def __call__(self, input):
    """
    Args:
      input: batch_size x 128 x 128 x 3
    Returns:
      output: 1D (1 if real, 0 if fake)
    """
    with tf.variable_scope(self.name):
      C64 = ops.Ck(input, 64, reuse=self.reuse, use_batchnorm=False, name='C64') # (?, 64, 64, 64)
      C128 = ops.Ck(C64, 128, reuse=self.reuse, name='C128')                     # (?, 32, 32, 128)
      C256 = ops.Ck(C128, 256, reuse=self.reuse, name='C256')                    # (?, 16, 16, 256)
      C512 = ops.Ck(C256, 512, reuse=self.reuse, name='C512')                    # (?, 8, 8, 512)

      # apply a convolution to produce a 1 dimensional output (1 channel?)
      # TODO: don't use leakyReLU and batch norm here
      output = tf.sigmoid(ops.Ck(C512, 1, reuse=self.reuse, stride=1, name='C1'), name='output')        # (?, 8, 8, 1)

    self.reuse = True
    self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    return output
