import tensorflow as tf
import ops
from discriminator import Discriminator
from generator import Generator

class CycleGAN:
  def __init__(self, Lambda=10):
    self.Lambda = Lambda
    self.G = Generator('G')
    self.D_Y = Discriminator('D_Y')
    self.F = Generator('F')
    self.D_X = Discriminator('D_X')

  def model(self, x, y):
    # cycle consistency loss (L1 norm)
    cycle_loss = tf.reduce_mean(tf.abs(self.F(self.G(x))-x)) + \
        tf.reduce_mean(tf.abs(self.G(self.F(y))-y))

    # G generator loss (Heuristic, non-saturating)
    G_loss = -tf.reduce_mean(ops.safe_log(self.D_Y(self.G(x)))) / 2 + \
        self.Lambda*cycle_loss

    # G discriminator loss
    D_Y_loss = (-tf.reduce_mean(ops.safe_log(self.D_Y(y))) - \
        tf.reduce_mean(ops.safe_log(1-self.D_Y(self.G(x))))) / 2

    tf.summary.scalar('D_Y/true', self.D_Y(y))
    tf.summary.scalar('D_Y/fake', self.D_Y(self.G(x)))

    # F generator loss (Heuristic, non-saturating)
    F_loss = -tf.reduce_mean(ops.safe_log(self.D_X(self.G(y)))) / 2 + \
        self.Lambda*cycle_loss

    # F discriminator loss
    D_X_loss = -(tf.reduce_mean(ops.safe_log(self.D_X(x))) - \
        tf.reduce_mean(ops.safe_log(1-self.D_X(self.F(y))))) / 2

    tf.summary.scalar('D_X/true', self.D_X(x))
    tf.summary.scalar('D_X/fake', self.D_X(self.F(y)))

    # total_loss = G_loss + F_loss + self.Lambda*cycle_loss
    tf.summary.scalar('loss/G', G_loss)
    tf.summary.scalar('loss/D_Y', D_Y_loss)
    tf.summary.scalar('loss/F', F_loss)
    tf.summary.scalar('loss/D_X', D_X_loss)
    tf.summary.scalar('loss/cycle', cycle_loss)

    tf.summary.image('X/origin', x)
    tf.summary.image('X/generated', self.G(x))
    tf.summary.image('X/reconstruction', self.F(self.G(x)))

    tf.summary.image('Y/origin', y)
    tf.summary.image('Y/generated', self.F(y))
    tf.summary.image('Y/reconstruction', self.G(self.F(y)))

    summary_op = tf.summary.merge_all()

    # generated x, reconstruction x, generated y, reconstruction y,
    # generated = [self.G(x), self.F(self.G(x)), self.F(y), self.G(self.F(y))]

    return G_loss, D_Y_loss, F_loss, D_X_loss, summary_op

  def optimize(self, G_loss, D_Y_loss, F_loss, D_X_loss):
    optimizer = tf.train.AdamOptimizer(learning_rate=2e-4)

    G_minimizer = optimizer.minimize(G_loss, var_list=self.G.variables)
    D_Y_minimizer = optimizer.minimize(D_Y_loss, var_list=self.D_Y.variables)
    F_minimizer = optimizer.minimize(F_loss, var_list=self.F.variables)
    D_X_minimizer = optimizer.minimize(D_X_loss, var_list=self.D_X.variables)

    with tf.control_dependencies([G_minimizer, D_Y_minimizer, F_minimizer, D_X_minimizer]):
      return tf.no_op(name='train')