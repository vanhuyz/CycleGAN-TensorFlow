import tensorflow as tf
from model import CycleGAN
from reader import Reader
from datetime import datetime
import os

X_TRAIN_FILE = 'data/tfrecords/apple.tfrecords'
Y_TRAIN_FILE = 'data/tfrecords/orange.tfrecords'

BATCH_SIZE = 1

def train():
  current_time = datetime.now().strftime("%Y%m%d-%H%M")
  checkpoints_dir = "checkpoints/{}".format(current_time)
  os.makedirs(checkpoints_dir, exist_ok=True)

  graph = tf.Graph()

  with graph.as_default():
    cycle_gan = CycleGAN(use_lsgan=True)

    X_reader = Reader(X_TRAIN_FILE, batch_size=BATCH_SIZE, name='X')
    Y_reader = Reader(Y_TRAIN_FILE, batch_size=BATCH_SIZE, name='Y')

    x = X_reader.feed()
    y = Y_reader.feed()

    G_loss, D_Y_loss, F_loss, D_X_loss, summary_op = cycle_gan.model(x, y)
    optimizer = cycle_gan.optimize(G_loss, D_Y_loss, F_loss, D_X_loss)

    saver = tf.train.Saver()

  train_writer = tf.summary.FileWriter(checkpoints_dir, graph)

  with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
      step = 0
      while not coord.should_stop():
        _, G_loss_val, D_Y_loss_val, F_loss_val, D_X_loss_val, summary = \
            sess.run([optimizer, G_loss, D_Y_loss, F_loss, D_X_loss, summary_op])

        train_writer.add_summary(summary, step)
        train_writer.flush()

        if step % 100 == 0:
          print('-----------Step %d:-------------' % step)
          print('  G_loss   : {}'.format(G_loss_val))
          print('  D_Y_loss   : {}'.format(D_Y_loss_val))
          print('  F_loss   : {}'.format(F_loss_val))
          print('  D_X_loss   : {}'.format(D_X_loss_val))

        if step % 1000 == 0:
          save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
          print("Model saved in file: %s" % save_path)

        step += 1

    except KeyboardInterrupt:
      print('Interrupted')
      coord.request_stop()
    except Exception as e:
      coord.request_stop(e)
    finally:
      save_path = saver.save(sess, checkpoints_dir + "/model.ckpt")
      print("Model saved in file: %s" % save_path)
      # When done, ask the threads to stop.
      coord.request_stop()
      coord.join(threads)

if __name__ == '__main__':
  train()
