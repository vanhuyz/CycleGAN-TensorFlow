import tensorflow as tf
import ops
import utils
from reader import Reader
from discriminator import Discriminator
from generator import Generator

REAL_LABEL = 1.0

class CycleGAN:
  def __init__(self,
               X_train_file='',
               Y_train_file='',
               batch_size=1,
               image_size=256,
               use_lsgan=True,
               norm='instance',
               lambda1=10,
               lambda2=10,
               learning_rate=2e-4,
               beta1=0.5,
               ngf=64
              ):
    """
    Args:
      X_train_file: string, X tfrecords file for training
      Y_train_file: string Y tfrecords file for training
      batch_size: integer, batch size
      image_size: integer, image size
      lambda1: integer, weight for forward cycle loss (X->Y->X)
      lambda2: integer, weight for backward cycle loss (Y->X->Y)
      use_lsgan: boolean
      norm: 'instance' or 'batch'
      learning_rate: float, initial learning rate for Adam
      beta1: float, momentum term of Adam
      ngf: number of gen filters in first conv layer
    """
    self.lambda1 = lambda1
    self.lambda2 = lambda2
    self.use_lsgan = use_lsgan
    use_sigmoid = not use_lsgan
    self.batch_size = batch_size
    self.image_size = image_size
    self.learning_rate = learning_rate
    self.beta1 = beta1
    self.X_train_file = X_train_file
    self.Y_train_file = Y_train_file

    self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')

    self.G = Generator('G', self.is_training, ngf=ngf, norm=norm, image_size=image_size)
    self.D_Y = Discriminator('D_Y',
        self.is_training, norm=norm, use_sigmoid=use_sigmoid)
    self.F = Generator('F', self.is_training, norm=norm, image_size=image_size)
    self.D_X = Discriminator('D_X',
        self.is_training, norm=norm, use_sigmoid=use_sigmoid)

    self.fake_x = tf.placeholder(tf.float32,
        shape=[batch_size, image_size, image_size, 3])
    self.fake_y = tf.placeholder(tf.float32,
        shape=[batch_size, image_size, image_size, 3])

  def model(self):
    X_reader = Reader(self.X_train_file, name='X',
        image_size=self.image_size, batch_size=self.batch_size)
    Y_reader = Reader(self.Y_train_file, name='Y',
        image_size=self.image_size, batch_size=self.batch_size)

    x = X_reader.feed()
    y = Y_reader.feed()

    # cache varibles to make training faster
    fake_xy = self.G(x)         # x -> y
    fake_xyx = self.F(fake_xy)  # x -> y -> x
    fake_yx = self.F(y)         # y -> x
    fake_yxy = self.G(fake_yx)  # y -> x -> y

    disc_Y_y = self.D_Y(y)
    disc_Y_xy = self.D_Y(fake_xy)
    disc_X_x = self.D_X(x)
    disc_X_yx = self.D_X(fake_yx)

    disc_X_fake_x = self.D_X(self.fake_x)
    disc_Y_fake_y = self.D_Y(self.fake_y)

    cycle_loss = self._cycle_consistency_loss(fake_xyx, fake_yxy, x, y)

    # X -> Y
    G_gan_loss = self._generator_loss(disc_Y_xy , use_lsgan=self.use_lsgan)
    G_loss =  G_gan_loss + cycle_loss
    D_Y_loss = self._discriminator_loss(disc_Y_y, disc_Y_fake_y, use_lsgan=self.use_lsgan)

    # Y -> X
    F_gan_loss = self._generator_loss(disc_X_yx, use_lsgan=self.use_lsgan)
    F_loss = F_gan_loss + cycle_loss
    D_X_loss = self._discriminator_loss(disc_X_x, disc_X_fake_x, use_lsgan=self.use_lsgan)

    # summary
    tf.summary.histogram('D_Y/true', disc_Y_y)
    tf.summary.histogram('D_Y/fake', disc_Y_xy)
    tf.summary.histogram('D_X/true', disc_X_x)
    tf.summary.histogram('D_X/fake', disc_X_yx)

    tf.summary.scalar('loss/G', G_gan_loss)
    tf.summary.scalar('loss/D_Y', D_Y_loss)
    tf.summary.scalar('loss/F', F_gan_loss)
    tf.summary.scalar('loss/D_X', D_X_loss)
    tf.summary.scalar('loss/cycle', cycle_loss)

    tf.summary.image('X/generated', utils.batch_convert2int(fake_xy))
    tf.summary.image('X/reconstruction', utils.batch_convert2int(fake_xyx))
    tf.summary.image('Y/generated', utils.batch_convert2int(fake_yx))
    tf.summary.image('Y/reconstruction', utils.batch_convert2int(fake_yxy))

    return G_loss, D_Y_loss, F_loss, D_X_loss, fake_xy, fake_yx

  def optimize(self, G_loss, D_Y_loss, F_loss, D_X_loss):
    def make_optimizer(loss, variables, name='Adam'):
      """ Adam optimizer with learning rate 0.0002 for the first 100k steps (~100 epochs)
          and a linearly decaying rate that goes to zero over the next 100k steps
      """
      global_step = tf.Variable(0, trainable=False)
      starter_learning_rate = self.learning_rate
      end_learning_rate = 0.0
      start_decay_step = 100000
      decay_steps = 100000
      beta1 = self.beta1
      learning_rate = (
          tf.where(
                  tf.greater_equal(global_step, start_decay_step),
                  tf.train.polynomial_decay(starter_learning_rate, global_step-start_decay_step,
                                            decay_steps, end_learning_rate,
                                            power=1.0),
                  starter_learning_rate
          )

      )
      tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)

      learning_step = (
          tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)
                  .minimize(loss, global_step=global_step, var_list=variables)
      )
      return learning_step

    G_optimizer = make_optimizer(G_loss, self.G.variables, name='Adam_G')
    D_Y_optimizer = make_optimizer(D_Y_loss, self.D_Y.variables, name='Adam_D_Y')
    F_optimizer =  make_optimizer(F_loss, self.F.variables, name='Adam_F')
    D_X_optimizer = make_optimizer(D_X_loss, self.D_X.variables, name='Adam_D_X')

    with tf.control_dependencies([G_optimizer, D_Y_optimizer, F_optimizer, D_X_optimizer]):
      return tf.no_op(name='optimizers')

  def _discriminator_loss(self, disc_real, disc_fake, use_lsgan=True):
    """
    Args:
      disc_real: output of discriminator when input is a real image (~ 1.0)
      disc_fake: output of discriminator when input is a fake image (~ 0.0)
    Note: default: D(y).shape == (batch_size,5,5,1),
                       fake_buffer_size=50, batch_size=1
    Args:
      G: generator object
      D: discriminator object
      y: 4D tensor (batch_size, image_size, image_size, 3)
    Returns:
      loss: scalar
    """
    if use_lsgan:
      # use mean squared error
      error_real = tf.reduce_mean(tf.squared_difference(disc_real, REAL_LABEL)) # disc_real ~ 1.0
      error_fake = tf.reduce_mean(tf.square(disc_fake))                         # disc_fake ~ 0.0
    else:
      # use cross entropy
      error_real = -tf.reduce_mean(ops.safe_log(disc_real))
      error_fake = -tf.reduce_mean(ops.safe_log(1-disc_fake))
    loss = (error_real + error_fake) / 2
    return loss

  def _generator_loss(self, disc_fake, use_lsgan=True):
    """  fool discriminator into believing that G(x) is real
    """
    if use_lsgan:
      # use mean squared error
      loss = tf.reduce_mean(tf.squared_difference(disc_fake, REAL_LABEL))
    else:
      # heuristic, non-saturating loss
      loss = -tf.reduce_mean(ops.safe_log(disc_fake)) / 2
    return loss

  def _cycle_consistency_loss(self, fake_xyx, fake_yxy, x, y):
    """ cycle consistency loss (L1 norm)
    """
    forward_loss = tf.reduce_mean(tf.abs(fake_xyx-x))
    backward_loss = tf.reduce_mean(tf.abs(fake_yxy-y))
    loss = self.lambda1*forward_loss + self.lambda2*backward_loss
    return loss
