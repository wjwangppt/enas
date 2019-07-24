from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import sys

import numpy as np
import tensorflow as tf

from src.utils import Logger
from src.utils import DEFINE_boolean
from src.utils import DEFINE_integer
from src.utils import DEFINE_string

from src.cnn.data_utils import read_data
from src.cnn.train import cnn_trainer
from src.rnn.search import search_train
from src.rnn.fixed import fixed_train

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

flags = tf.app.flags
gfile = tf.gfile
FLAGS = flags.FLAGS

DEFINE_string("network_type", "cnn","")
DEFINE_boolean("reset_output_dir", False, "Delete output_dir if exists.")
DEFINE_string("data_path", "", "")
DEFINE_string("output_dir", "", "")
DEFINE_string("data_format", "NHWC", "'NHWC' or 'NCWH'")
DEFINE_string("search_for", None, "Must be [macro|micro]")
DEFINE_boolean("controller_training", False, "")

DEFINE_integer("batch_size", 128, "")
DEFINE_string("fixed_arc", None, "")
DEFINE_integer("num_epochs", 300, "")

DEFINE_integer("log_every", 50, "How many steps to log")
DEFINE_integer("eval_every_epochs", 1, "How many epochs to eval")


def main(_):

  print("-" * 80)
  if not gfile.IsDirectory(FLAGS.output_dir):
    print('Path {} does not exist. Creating'.format(FLAGS.output_dir))
    gfile.MakeDirs(FLAGS.output_dir)
  elif FLAGS.reset_output_dir:
    print('Path {} exists. Reseting'.format(FLAGS.output_dir))
    gfile.DeleteRecursively(FLAGS.output_dir)
    gfile.MakeDirs(FLAGS.output_dir)

  print("-" * 80)
  log_file = os.path.join(FLAGS.output_dir, "stdout")
  print("Logging to {}".format(log_file))
  sys.stdout = Logger(log_file)

  #utils.print_user_flags()

  params = tf.contrib.training.HParams(
      data_path=FLAGS.data_path,
      log_every=FLAGS.log_every,
      output_dir=FLAGS.output_dir,
      batch_size=FLAGS.batch_size,
      num_epochs=FLAGS.num_epochs,
      search_for=FLAGS.search_for,
      controller_training=FLAGS.controller_training
  )

  if FLAGS.network_type == "cnn":
    if FLAGS.fixed_arc is None:
      images, labels = read_data(FLAGS.data_path)
    else:
      images, labels = read_data(FLAGS.data_path, num_valids=0)
    cnn_trainer(params, images, labels)
  elif FLAGS.network_type == "rnn":
    with gfile.GFile(params.data_path, 'rb') as finp:
      x_train, x_valid, x_test, _, _ = pickle.load(finp)
      print('-' * 80)
      print('train_size: {0}'.format(np.size(x_train)))
      print('valid_size: {0}'.format(np.size(x_valid)))
      print(' test_size: {0}'.format(np.size(x_test)))
    if FLAGS.fixed_arc != None:
      fixed_train(params, x_train, x_valid, x_test)
    else:
      search_train(params, x_train, x_valid)

if __name__ == "__main__":
  print(sys.getdefaultencoding())
  tf.app.run()
