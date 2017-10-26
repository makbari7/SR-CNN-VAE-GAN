import os
import scipy.misc
import numpy as np

from model import VAEGAN
from utils import pp, to_json, show_all_variables

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 500, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0001, "default learning rate of for adam [0.0001]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train samples [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch samples [64]")
flags.DEFINE_integer("input_height", 88, "The size of sample to use.")
flags.DEFINE_integer("input_width", 16, "The size of sample to use. If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 88, "The size of the output samples to produce [64]")
flags.DEFINE_integer("output_width", 16, "The size of the output samples to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "Nottingham", "The name of dataset.")
flags.DEFINE_string("input_fname_pattern", "*.mid", "Glob pattern of filename of inputs [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the sample samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
FLAGS = flags.FLAGS

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if FLAGS.input_width is None:
    FLAGS.input_width = FLAGS.input_height
  if FLAGS.output_width is None:
    FLAGS.output_width = FLAGS.output_height

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True

  with tf.Session(config=run_config) as sess:
    vaegan = VAEGAN(
        sess,
        input_width=FLAGS.input_width,
        input_height=FLAGS.input_height,
        output_width=FLAGS.output_width,
        output_height=FLAGS.output_height,
        batch_size=FLAGS.batch_size,
        sample_num=FLAGS.batch_size,
        dataset_name=FLAGS.dataset,
        input_fname_pattern=FLAGS.input_fname_pattern,        
        checkpoint_dir=FLAGS.checkpoint_dir,
        sample_dir=FLAGS.sample_dir)

    show_all_variables()

    if FLAGS.train:
      vaegan.train(FLAGS)
    else:
      if not vaegan.load(FLAGS.checkpoint_dir)[0]:
        raise Exception("[!] Train a model first, then run test mode")
      else:
        vaegan.test(FLAGS)
      
if __name__ == '__main__':
  tf.app.run()
