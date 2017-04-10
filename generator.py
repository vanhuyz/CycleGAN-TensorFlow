import tensorflow as tf
import ops

class Generator:
  def __init__(self):
    pass

  def __call__(self, input):
    """
    Args:
      input: batch_size x 128 x 128 x 3
    Returns:
      output: same size as input
    """

    # convolution
    c7s1_32 = ops.c7s1_k(input, 32, name='c7s1_32') # (?, 128, 128, 32)
    d64 = ops.dk(c7s1_32, 64, name='d64')           # (?, 64, 64, 64)
    d128 = ops.dk(d64, 128, name='d128')            # (?, 32, 32, 128)

    # 6 residual blocks
    R128_1 = ops.Rk(d128, 128, name='R128_1')       # (?, 32, 32, 128)
    R128_2 = ops.Rk(R128_1, 128, name='R128_2')     # (?, 32, 32, 128)
    R128_3 = ops.Rk(R128_2, 128, name='R128_3')     # (?, 32, 32, 128)
    R128_4 = ops.Rk(R128_3, 128, name='R128_4')     # (?, 32, 32, 128)
    R128_5 = ops.Rk(R128_4, 128, name='R128_5')     # (?, 32, 32, 128)
    R128_6 = ops.Rk(R128_5, 128, name='R128_6')     # (?, 32, 32, 128)

    # fractional-strided convolution
    u64 = ops.uk(R128_6, 64, name='u64')            # (?, 64, 64, 64)
    u32 = ops.uk(u64, 32, name='u32')               # (?, 128, 128, 32)

    # convolution
    output = ops.c7s1_k(u32, 3, name='output')      # (?, 128, 128, 3)

    return output