import tensorflow as tf
from model import CycleGAN
from reader import Reader

X_TRAIN_FILE = 'data/tfrecords/apple.tfrecords'
Y_TRAIN_FILE = 'data/tfrecords/orange.tfrecords'

BATCH_SIZE = 1

def train():
  graph = tf.Graph()
  cycle_gan = CycleGAN()

  with graph.as_default():
    X_reader = Reader(X_TRAIN_FILE, batch_size=BATCH_SIZE)
    Y_reader = Reader(Y_TRAIN_FILE, batch_size=BATCH_SIZE)

    x = X_reader.feed()
    y = Y_reader.feed()

    loss_op = cycle_gan.loss(x, y)
    optimizer = cycle_gan.optimize(loss_op)

  with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
      step = 0
      while not coord.should_stop():
        _, loss = sess.run([optimizer, loss_op])

        print('-----------Step %d:-------------' % step)
        print('  Loss   : {}'.format(loss))

        step += 1

    except KeyboardInterrupt:
      print('Interrupted')
      coord.request_stop()
    except Exception as e:
      coord.request_stop(e)
    finally:
      # save_path = saver.save(sess, checkpoints_dir + "/model.ckpt")
      # print("Model saved in file: %s" % save_path)
      # When done, ask the threads to stop.
      coord.request_stop()
      coord.join(threads)

if __name__ == '__main__':
  train()
