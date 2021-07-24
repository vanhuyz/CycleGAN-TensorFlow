""" Freeze variables and convert 2 generator networks to 2 GraphDef files.
This makes file size smaller and can be used for inference in production.
An example of command-line usage is:
python export_graph.py --checkpoint_dir checkpoints/20170424-1152 \
                       --XtoY_model ukiyoe2photo.pb \
                       --YtoX_model photo2ukiyoe.pb \
                       --image_size 256
"""

import tensorflow as tf
import os
from tensorflow.python.tools.freeze_graph import freeze_graph
from model import CycleGAN
import utils

FLAGS = tf.compat.v1.flags.FLAGS

tf.compat.v1.flags.DEFINE_string('checkpoint_dir', '', 'checkpoints directory path')
tf.compat.v1.flags.DEFINE_string('XtoY_model', 'ukiyoe2photo.pb', 'XtoY model name, default: ukiyoe2photo.pb')
tf.compat.v1.flags.DEFINE_string('YtoX_model', 'photo2ukiyoe.pb', 'YtoX model name, default: photo2ukiyoe.pb')
tf.compat.v1.flags.DEFINE_integer('image_size', '256', 'image size, default: 256')
tf.compat.v1.flags.DEFINE_integer('ngf', 64,
                        'number of gen filters in first conv layer, default: 64')
tf.compat.v1.flags.DEFINE_string('norm', 'instance',
                       '[instance, batch] use instance norm or batch norm, default: instance')


def export_graph(model_name, XtoY=True):
    graph = tf.Graph()

    with graph.as_default():
        cycle_gan = CycleGAN(ngf=FLAGS.ngf, norm=FLAGS.norm, image_size=FLAGS.image_size)

        input_image = tf.compat.v1.placeholder(tf.float32, shape=[FLAGS.image_size, FLAGS.image_size, 3], name='input_image')
        cycle_gan.model()
        if XtoY:
            output_image = cycle_gan.G.sample(tf.expand_dims(input_image, 0))
        else:
            output_image = cycle_gan.F.sample(tf.expand_dims(input_image, 0))

        output_image = tf.identity(output_image, name='output_image')
        restore_saver = tf.compat.v1.train.Saver()
        export_saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session(graph=graph) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        latest_ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        restore_saver.restore(sess, latest_ckpt)
        output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess, graph.as_graph_def(), [output_image.op.name])

        tf.compat.v1.train.write_graph(output_graph_def, 'pretrained', model_name, as_text=False)


def main(unused_argv):
    print('Export XtoY model...')
    export_graph(FLAGS.XtoY_model, XtoY=True)
    print('Export YtoX model...')
    export_graph(FLAGS.YtoX_model, XtoY=False)


if __name__ == '__main__':
    tf.compat.v1.app.run()
