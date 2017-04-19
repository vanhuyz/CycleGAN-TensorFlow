import tensorflow as tf
import ops

class Discriminator:
  def __init__(self, name, is_training, norm='instance', patch_size=70, use_sigmoid=False):
    self.name = name
    self.is_training = is_training
    self.norm = norm
    self.reuse = False
    self.patch_size = 70
    self.use_sigmoid = use_sigmoid

  def __call__(self, input):
    """
    Args:
      input: batch_size x image_size x image_size x 3
    Returns:
      output: 4D tensor batch_size x out_size x out_size x 1 (default 1x5x5x1)
              filled with 0.9 if real, 0.0 if fake
    """
    with tf.variable_scope(self.name):
      batch_size = input.get_shape().as_list()[0]
      patch = tf.random_crop(input,
          [batch_size, self.patch_size, self.patch_size, 3])    # (?, 70, 70, 70)

      # convolution layers
      C64 = ops.Ck(patch, 64, reuse=self.reuse, norm=None,
          is_training=self.is_training, name='C64')             # (?, 35, 35, 64)
      C128 = ops.Ck(C64, 128, reuse=self.reuse, norm=self.norm,
          is_training=self.is_training, name='C128')            # (?, 18, 18, 128)
      C256 = ops.Ck(C128, 256, reuse=self.reuse, norm=self.norm,
          is_training=self.is_training, name='C256')            # (?, 9, 9, 256)
      C512 = ops.Ck(C256, 512,reuse=self.reuse, norm=self.norm,
          is_training=self.is_training, name='C512')            # (?, 5, 5, 512)

      # apply a convolution to produce a 1 dimensional output (1 channel?)
      # use_sigmoid = False if use_lsgan = True
      output = ops.last_conv(C512, reuse=self.reuse,
          use_sigmoid=self.use_sigmoid, name='output')          # (?, 5, 5, 1)

    self.reuse = True
    self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    return output
