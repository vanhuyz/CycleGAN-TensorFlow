import tensorflow as tf

def c7s1_k(input, k, reuse=False, is_training=True, name=None):
  """ A 7x7 Convolution-BatchNorm-ReLU layer with k filters and stride 1
  Args:
    input: 4D tensor
    k: integer, number of filters (output depth)
    name: string, e.g. 'c7sk-32'
  Returns:
    4D tensor
  """
  with tf.variable_scope(name, reuse=reuse):
    weights = _weights("weights",
      shape=[7, 7, input.get_shape()[3], k])
    biases = tf.get_variable("biases", [k],
        initializer=tf.constant_initializer(0.0))

    # TODO: reflection padding
    conv = tf.nn.conv2d(input, weights,
        strides=[1, 1, 1, 1], padding='SAME')
    bn = batch_norm(conv+biases, is_training)
    output = tf.nn.relu(bn)
    return output

def dk(input, k, reuse=False, is_training=True, name=None):
  """ A 3x3 Convolution-BatchNorm-ReLU layer with k filters and stride 2
  Args:
    input: 4D tensor
    k: integer, number of filters (output depth)
    name: string, e.g. 'd64'
  Returns:
    4D tensor
  """
  with tf.variable_scope(name, reuse=reuse):
    weights = _weights("weights",
      shape=[3, 3, input.get_shape()[3], k])
    biases = tf.get_variable("biases", [k],
        initializer=tf.constant_initializer(0.0))

    # TODO: reflection padding
    conv = tf.nn.conv2d(input, weights,
        strides=[1, 2, 2, 1], padding='SAME')
    bn = batch_norm(conv+biases, is_training)
    output = tf.nn.relu(bn)
    return output

def Rk(input, k, reuse=False, name=None):
  """ A residual block that contains two 3x3 convolutional layers
      with the same number of filters on both layer
  """
  with tf.variable_scope(name):
    # layer 1
    weights1 = _weights("weights1",
      shape=[3, 3, input.get_shape()[3], k])
    biases1 = tf.get_variable("biases1", [k],
        initializer=tf.constant_initializer(0.0))
    conv1 = tf.nn.conv2d(input, weights1,
        strides=[1, 1, 1, 1], padding='SAME')
    relu1 = tf.nn.relu(conv1+biases1)

    # layer 2
    weights2 = _weights("weights2",
      shape=[3, 3, relu1.get_shape()[3], k])
    biases2 = tf.get_variable("biases2", [k],
        initializer=tf.constant_initializer(0.0))
    conv2 = tf.nn.conv2d(relu1, weights1,
        strides=[1, 1, 1, 1], padding='SAME')
    res = conv2+biases2
    relu2 = tf.nn.relu(input+res)
    return relu2

def uk(input, k, reuse=False, is_training=True, name=None):
  """ A 3x3 fractional-strided-Convolution-BatchNorm-ReLU layer with k filters, stride 1/2
  Args:
    input: 4D tensor
    k: integer, number of filters (output depth)
    name: string, e.g. 'c7sk-32'
  Returns:
    4D tensor
  """
  with tf.variable_scope(name, reuse=reuse):
    weights = _weights("weights",
      shape=[3, 3, input.get_shape()[3], k])
    biases = tf.get_variable("biases", [k],
        initializer=tf.constant_initializer(0.0))

    # TODO: reflection padding
    fsconv = tf.nn.conv2d_transpose(input, weights,
        strides=[1, 2, 2, 1], padding='SAME')
    bn = batch_norm(fsconv+biases, is_training)
    output = tf.nn.relu(bn)
    return output

def batch_norm(input, is_training):
  """ TODO: set hyper-parameter
  """
  return tf.contrib.layers.batch_norm(input, decay=0.9, is_training=is_training)

def _weights(name, shape, mean=0.0, stddev=0.02):
  """ Helper to create an initialized Variable
  Args:
    name: name of the variable
    shape: list of ints
    mean: mean of a Gaussian
    stddev: standard deviation of a Gaussian
  """
  var = tf.get_variable(
    name,
    shape,
    initializer=tf.random_normal_initializer(mean=mean, stddev=stddev, dtype=tf.float32))
  return var
