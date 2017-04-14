import tensorflow as tf
import ops
import utils
from reader import Reader
from discriminator import Discriminator
from generator import Generator

X_TRAIN_FILE = 'data/tfrecords/apple.tfrecords'
Y_TRAIN_FILE = 'data/tfrecords/orange.tfrecords'
REAL_LABEL = 0.9

class CycleGAN:
  def __init__(self, batch_size=1,
    image_size=128, use_lsgan=True,
    lambda1=10, lambda2=10):
    """
    Args:
      lambda1: integer, forward cycle loss weight
      lambda2: integer, backward cycle loss weight
      use_lsgan: boolean
    """
    self.lambda1 = lambda1
    self.lambda2 = lambda2
    self.use_lsgan = use_lsgan
    use_sigmoid = not use_lsgan

    self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')

    self.G = Generator('G', self.is_training)
    self.D_Y = Discriminator('D_Y', self.is_training, use_sigmoid=use_sigmoid)
    self.F = Generator('F', self.is_training)
    self.D_X = Discriminator('D_X', self.is_training, use_sigmoid=use_sigmoid)

    X_reader = Reader(X_TRAIN_FILE, name='X')
    Y_reader = Reader(Y_TRAIN_FILE, name='Y')

    self.x = X_reader.feed()
    self.y = Y_reader.feed()

  def model(self):
    cycle_loss = self.cycle_consistency_loss(self.G, self.F, self.x, self.y)

    # X -> Y
    G_gan_loss = self.generator_loss(self.G, self.D_Y, self.x, use_lsgan=self.use_lsgan)
    G_loss =  G_gan_loss + cycle_loss
    D_Y_loss = self.discriminator_loss(self.G, self.D_Y, self.x, self.y, use_lsgan=self.use_lsgan)

    # Y -> X
    F_gan_loss = self.generator_loss(self.F, self.D_X, self.y, use_lsgan=self.use_lsgan)
    F_loss = F_gan_loss + cycle_loss
    D_X_loss = self.discriminator_loss(self.F, self.D_X, self.y, self.x, use_lsgan=self.use_lsgan)

    # summary
    tf.summary.histogram('D_Y/true', self.D_Y(self.y))
    tf.summary.histogram('D_Y/fake', self.D_Y(self.G(self.x)))
    tf.summary.histogram('D_X/true', self.D_X(self.x))
    tf.summary.histogram('D_X/fake', self.D_X(self.F(self.y)))

    tf.summary.scalar('loss/G', G_gan_loss)
    tf.summary.scalar('loss/D_Y', D_Y_loss)
    tf.summary.scalar('loss/F', F_gan_loss)
    tf.summary.scalar('loss/D_X', D_X_loss)
    tf.summary.scalar('loss/cycle', cycle_loss)

    tf.summary.image('X/generated', utils.batch_convert2int(self.G(self.x)))
    tf.summary.image('X/reconstruction', utils.batch_convert2int(self.F(self.G(self.x))))
    tf.summary.image('Y/generated', utils.batch_convert2int(self.F(self.y)))
    tf.summary.image('Y/reconstruction', utils.batch_convert2int(self.G(self.F(self.y))))

    self.summary = tf.summary.merge_all()
    self.saver = tf.train.Saver()

    return G_loss, D_Y_loss, F_loss, D_X_loss

  def optimize(self, G_loss, D_Y_loss, F_loss, D_X_loss):
    def make_optimizer(loss, variables, name='Adam'):
      """ Adam optimizer with learning rate 0.0002 for the first 100k steps (~100 epochs)
          and a linearly decaying rate that goes to zero over the next 100k steps
          Note: use Adam with learning rate decay -> seems strange?
                Not sure how to implement this in TensorFlow.
                Here is AdamOptimizer with a linearly decaying rate
                goes to zero after 200k steps
      """
      global_step = tf.Variable(0, trainable=False)
      starter_learning_rate = 2e-4
      end_learning_rate = 0.0
      decay_steps = 200000

      learning_rate = (
          tf.train.polynomial_decay(starter_learning_rate, global_step,
                                    decay_steps, end_learning_rate,
                                    power=1.0)
      )

      learning_step = (
          tf.train.AdamOptimizer(learning_rate, name=name)
                  .minimize(loss, global_step=global_step, var_list=variables)
      )
      return learning_step

    G_optimizer = make_optimizer(G_loss, self.G.variables, name='Adam_G')
    D_Y_optimizer = make_optimizer(D_Y_loss, self.D_Y.variables, name='Adam_D_Y')
    F_optimizer =  make_optimizer(F_loss, self.F.variables, name='Adam_F')
    D_X_optimizer = make_optimizer(D_X_loss, self.D_X.variables, name='Adam_D_X')

    with tf.control_dependencies([G_optimizer, D_Y_optimizer, F_optimizer, D_X_optimizer]):
      return tf.no_op(name='optimizers')

  def discriminator_loss(self, G, D, x, y, use_lsgan=True):
    """ Note: D(y).shape == (batch_size,8,8,1),
              default: fake_buffer_size=50, batch_size=1
    Args:
      G: generator object
      D: discriminator object
      y: 4D tensor (batch_size, image_size, image_size, 3)
    Returns:
      loss: scalar
    """
    if use_lsgan:
      # use mean squared error
      error_real = tf.reduce_mean(tf.squared_difference(D(y), REAL_LABEL))
      error_fake = tf.reduce_mean(tf.square(D(G(x))))
    else:
      # use cross entropy
      error_real = -tf.reduce_mean(ops.safe_log(D(y)))
      error_fake = -tf.reduce_mean(ops.safe_log(1-D(G(x))))
    loss = (error_real + error_fake) / 2
    return loss

  def generator_loss(self, G, D, x, use_lsgan=True):
    """  fool discriminator into believing that G(x) is real
    """
    if use_lsgan:
      # use mean squared error
      loss = tf.reduce_mean(tf.squared_difference(D(G(x)), REAL_LABEL))
    else:
      # heuristic, non-saturating loss
      loss = -tf.reduce_mean(ops.safe_log(D(G(x)))) / 2
    return loss

  def cycle_consistency_loss(self, F, G, x, y):
    """ cycle consistency loss (L1 norm)
    """
    forward_loss = tf.reduce_mean(tf.abs(F(G(x))-x))
    backward_loss = tf.reduce_mean(tf.abs(G(F(y))-y))
    loss = self.lambda1*forward_loss + self.lambda2*backward_loss
    return loss

  def sample(self, input, G_or_F='G'):
    if G_or_F == 'G':
      image = utils.batch_convert2int(self.G(input))
    else:
      image = utils.batch_convert2int(self.F(input))

    image = tf.image.encode_jpeg(tf.squeeze(image, [0]))
    return image
