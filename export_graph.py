import tensorflow as tf
import os
from tensorflow.python.tools.freeze_graph import freeze_graph
from model import CycleGAN
import utils

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('checkpoints_dir', '', 'checkpoints directory path')
tf.flags.DEFINE_string('XtoY_model', '', 'XtoY model name, e.g. apple2orange.pb')
tf.flags.DEFINE_string('YtoX_model', '', 'YtoX model name, e.g. orange2apple.pb')


def export_graph(model_name, XtoY=True):
  graph = tf.Graph()

  with graph.as_default():
    cycle_gan = CycleGAN(norm='instance')

    input_image = tf.placeholder(tf.float32, shape=[128, 128, 3], name='input_image')
    cycle_gan.model()
    if XtoY:
      output_image = cycle_gan.G.sample(tf.expand_dims(input_image, 0))
    else:
      output_image = cycle_gan.F.sample(tf.expand_dims(input_image, 0))

    output_image = tf.identity(output_image, name='output_image')
    restore_saver = tf.train.Saver()
    export_saver = tf.train.Saver()

  with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    latest_ckpt = tf.train.latest_checkpoint(FLAGS.checkpoints_dir)
    restore_saver.restore(sess, latest_ckpt)
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), [output_image.op.name])

    tf.train.write_graph(output_graph_def, 'pretrained', model_name, as_text=False)

def main(unused_argv):
  print('Export XtoY model...')
  export_graph(FLAGS.XtoY_model, XtoY=True)
  print('Export YtoX model...')
  export_graph(FLAGS.YtoX_model, XtoY=False)

if __name__ == '__main__':
  tf.app.run()
