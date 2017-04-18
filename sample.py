import tensorflow as tf
import os

from model import CycleGAN
import utils

CKPT_PATH = 'checkpoints/20170414-2228/model.ckpt'
IMG_PATH = 'data/apple2orange/testA/n07740461_10011.jpg'

def sample():
  graph = tf.Graph()

  with graph.as_default():
    cycle_gan = CycleGAN()

    with tf.gfile.FastGFile(IMG_PATH, 'r') as f:
      image_data = f.read()
    in_image = tf.image.decode_jpeg(image_data, channels=3)
    in_image = tf.image.resize_images(in_image, size=(128, 128))
    in_image = utils.convert2float(in_image)
    in_image.set_shape([128, 128, 3])

    cycle_gan = CycleGAN()
    cycle_gan.model()
    out_image = cycle_gan.G.sample(tf.expand_dims(in_image, 0))

  with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    cycle_gan.saver.restore(sess, CKPT_PATH)
    generated = out_image.eval()
    samples_dir = 'samples'
    os.makedirs(samples_dir, exist_ok=True)
    samples_file = os.path.join(samples_dir, 'sample.jpg')
    with open(samples_file, 'wb') as f:
      f.write(generated)

if __name__ == '__main__':
  sample()
