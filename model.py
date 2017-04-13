import tensorflow as tf
import ops
import utils
from discriminator import Discriminator
from generator import Generator

class CycleGAN:
  def __init__(self, batch_size=1, fake_buffer_size=50,
    image_size=128, use_lsgan=True, lambda1=10, lambda2=10):
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

    self.G = Generator('G')
    self.D_Y = Discriminator('D_Y', use_sigmoid=use_sigmoid)
    self.F = Generator('F')
    self.D_X = Discriminator('D_X', use_sigmoid=use_sigmoid)

    # buffer that stores generated images
    self.fake_buffer_G = tf.get_variable('fake_buffer_G',
        shape=[fake_buffer_size, image_size, image_size, 3],
        initializer=tf.constant_initializer(0.0),
        trainable=False)
    self.fake_buffer_F = tf.get_variable('fake_buffer_F',
        shape=[fake_buffer_size, image_size, image_size, 3],
        initializer=tf.constant_initializer(0.0),
        trainable=False)

  def update_fake_buffer(self, x, y):
    """ Keep image buffers that store the newest generated images
    """
    self.fake_buffer_G = tf.concat([self.fake_buffer_G[1:,:,:,:], self.G(x)], axis=0)
    self.fake_buffer_F = tf.concat([self.fake_buffer_F[1:,:,:,:], self.F(y)], axis=0)

  def discriminator_loss(self, G, D, fake_buffer, y, use_lsgan=True):
    """ note: D(y).shape == (batch_size,8,8,1), default fake_buffer_size=50, batch_size=1
    Seems unbalanced?
    Args:
      G: generator object
      D: discriminator object
      fake_buffer: 4D tensor (fake_buffer_size, image_size, image_size, 3)
      y: 4D tensor (batch_size, image_size, image_size, 3)
    Returns:
      loss: scalar
    """
    if use_lsgan:
      # use mean squared error
      error_real = tf.reduce_mean(tf.squared_difference(D(y),1))
      error_fake = tf.reduce_mean(tf.square(D(fake_buffer)))
    else:
      # use cross entropy
      error_real = -tf.reduce_mean(ops.safe_log(D(y)))
      error_fake = -tf.reduce_mean(ops.safe_log(1-D(fake_buffer)))
    loss = (error_real + error_fake) / 2
    return loss

  def generator_loss(self, G, D, x, use_lsgan=True):
    """ try to fool discriminator into believing that G(x) is real
    """
    if use_lsgan:
      # use mean squared error
      loss = tf.reduce_mean(tf.squared_difference(D(G(x)),1))
    else:
      # heuristic, non-saturating loss
      loss = -tf.reduce_mean(ops.safe_log(D(G(x)))) / 2
    return loss

  def cycle_consistency_loss(self, G, F, x, y):
    """ cycle consistency loss (L1 norm)
    """
    forward_loss = tf.reduce_mean(tf.abs(F(G(x))-x))
    backward_loss = tf.reduce_mean(tf.abs(G(F(y))-y))
    loss = self.lambda1*forward_loss + self.lambda2*backward_loss
    return loss

  def model(self, x, y):
    self.update_fake_buffer(x, y)
    cycle_loss = self.cycle_consistency_loss(self.G, self.F, x, y)

    # X -> Y
    G_gan_loss = self.generator_loss(self.G, self.D_Y, x, use_lsgan=self.use_lsgan)
    G_loss =  G_gan_loss + cycle_loss
    D_Y_loss = self.discriminator_loss(self.G, self.D_Y, self.fake_buffer_G, y, use_lsgan=self.use_lsgan)

    tf.summary.histogram('D_Y/true', self.D_Y(y))
    tf.summary.histogram('D_Y/fake', self.D_Y(self.G(x)))

    # Y -> X
    F_gan_loss = self.generator_loss(self.F, self.D_X, y, use_lsgan=self.use_lsgan)
    F_loss = F_gan_loss + cycle_loss
    D_X_loss = self.discriminator_loss(self.F, self.D_X, self.fake_buffer_F, x, use_lsgan=self.use_lsgan)

    # summary
    tf.summary.histogram('D_X/true', self.D_X(x))
    tf.summary.histogram('D_X/fake', self.D_X(self.F(y)))

    tf.summary.scalar('loss/G', G_gan_loss)
    tf.summary.scalar('loss/D_Y', D_Y_loss)
    tf.summary.scalar('loss/F', F_gan_loss)
    tf.summary.scalar('loss/D_X', D_X_loss)
    tf.summary.scalar('loss/cycle', cycle_loss)

    tf.summary.image('X/generated', utils.batch_convert2int(self.G(x)))
    tf.summary.image('X/reconstruction', utils.batch_convert2int(self.F(self.G(x))))
    tf.summary.image('Y/generated', utils.batch_convert2int(self.F(y)))
    tf.summary.image('Y/reconstruction', utils.batch_convert2int(self.G(self.F(y))))

    summary_op = tf.summary.merge_all()

    return G_loss, D_Y_loss, F_loss, D_X_loss, summary_op

  def optimize(self, G_loss, D_Y_loss, F_loss, D_X_loss):
    def make_optimizer(loss, variables, name='Adam'):
      """ Adam optimizer with learning rate 0.0002 for the first 100000 steps (~100 epochs)
          and a linearly decaying rate that goes to zero over the next 100000 steps
      """
      global_step = tf.Variable(0, trainable=False)
      starter_learning_rate = 2e-4
      end_learning_rate = 0.0
      without_decay_steps = 100000
      decay_steps = 100000

      learning_rate = tf.where(tf.greater(global_step, without_decay_steps),
          starter_learning_rate,
          tf.train.polynomial_decay(starter_learning_rate, global_step,
                                        decay_steps, end_learning_rate,
                                        power=1.0))

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
      return tf.no_op(name='train')
