import tensorflow as tf
import ops

class Generator:
  def __init__(self, name):
    self.name = name
    self.reuse = False

  def __call__(self, input):
    """
    Args:
      input: batch_size x 128 x 128 x 3
    Returns:
      output: same size as input
    """
    with tf.variable_scope(self.name):
      # convolution
      c7s1_32 = ops.c7s1_k(input, 32, reuse=self.reuse, name='c7s1_32') # (?, 128, 128, 32)
      d64 = ops.dk(c7s1_32, 64, reuse=self.reuse, name='d64')           # (?, 64, 64, 64)
      d128 = ops.dk(d64, 128, reuse=self.reuse, name='d128')            # (?, 32, 32, 128)

      # 6 residual blocks
      R128_1 = ops.Rk(d128, 128, reuse=self.reuse, name='R128_1')       # (?, 32, 32, 128)
      R128_2 = ops.Rk(R128_1, 128, reuse=self.reuse, name='R128_2')     # (?, 32, 32, 128)
      R128_3 = ops.Rk(R128_2, 128, reuse=self.reuse, name='R128_3')     # (?, 32, 32, 128)
      R128_4 = ops.Rk(R128_3, 128, reuse=self.reuse, name='R128_4')     # (?, 32, 32, 128)
      R128_5 = ops.Rk(R128_4, 128, reuse=self.reuse, name='R128_5')     # (?, 32, 32, 128)
      R128_6 = ops.Rk(R128_5, 128, reuse=self.reuse, name='R128_6')     # (?, 32, 32, 128)

      # fractional-strided convolution
      u64 = ops.uk(R128_6, 64, reuse=self.reuse, name='u64')            # (?, 64, 64, 64)
      u32 = ops.uk(u64, 32, reuse=self.reuse, name='u32')               # (?, 128, 128, 32)

      # convolution
      output = ops.c7s1_k(u32, 3, reuse=self.reuse, name='output')      # (?, 128, 128, 3)

    # set reuse=True for next call
    self.reuse = True
    self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
    # print(self.variables)
    return output