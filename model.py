import tensorflow as tf
from discriminator import Discriminator
from generator import Generator

class CycleGAN:
  def __init__(self, Lambda=10):
    self.Lambda = Lambda
    self.G = Generator()
    self.D_Y = Discriminator()
    self.F = Generator()
    self.D_X = Discriminator()

  def loss(self, x, y):
    G_loss = tf.reduce_mean(tf.log(self.D_Y(y))) +
        tf.reduce_mean(tf.log(1-self.D_Y(self.G(x))))

    F_loss = tf.reduce_mean(tf.log(self.D_X(x))) +
        tf.reduce_mean(tf.log(1-self.D_X(self.F(y))))

    cycle_loss = tf.reduce_mean(tf.abs(self.F(self.G(x))-x)) +
        tf.reduce_mean(tf.abs(self.G(self.F(y))-y))

    total_loss = G_loss + F_loss + self.Lambda*cycle_loss
    return total_loss

  def optimize(self, loss):
    optimizer = tf.train.AdamOptimizer(learning_rate=2e-4)

    dis_variables = self.D_X.variables + self.D_Y.variables
    gen_variables = self.G.variables + self.F.variables

    dis_update = optimizer.minimize(-loss, var_list=dis_variables)
    gen_update = optimizer.minimize(loss, var_list=gen_variables)

    with tf.control_dependencies([dis_update, gen_update]):
      return tf.no_op(name='train')