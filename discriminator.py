import tensorflow as tf
import ops

class Discriminator:
  def __init__(self):
    pass

  def __call__(self, input):
    """
    Args:
      input: batch_size x 128 x 128 x 3
    Returns:
      output: 1D (1 if real, 0 if fake)
    """
    C64 = ops.Ck(input, 64, use_batchnorm=False, name='C64') # (?, 64, 64, 64)
    C128 = ops.Ck(C64, 128, name='C128')                     # (?, 32, 32, 128)
    C256 = ops.Ck(C128, 256, name='C256')                    # (?, 16, 16, 256)
    C512 = ops.Ck(C256, 512, name='C512')                    # (?, 8, 8, 512)

    # apply a convolution to produce a 1 dimensional output
    conv = ops.Ck(C512, 1, stride=8)                         # (?, 1, 1, 1)
    output = tf.squeeze(conv, name='output')                 # (?)
    return output
