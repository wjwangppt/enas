
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

from src.cnn.general_controller import GeneralController
from src.cnn.general_child import GeneralChild

from src.cnn.micro_controller import MicroController
from src.cnn.micro_child import MicroChild


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean("controller_search_whole_channels", True,"")
flags.DEFINE_boolean("controller_sync_replicas", True, "To sync or not to sync.")
flags.DEFINE_boolean("child_sync_replicas", False, "To sync or not to sync.")

def get_ops(params,images, labels):
  """
  Args:
    images: dict with keys {"train", "valid", "test"}.
    labels: dict with keys {"train", "valid", "test"}.
  """

  assert FLAGS.search_for is not None, "Please specify --search_for"

  if FLAGS.search_for == "micro": 
    ControllerClass = MicroController
    ChildClass = MicroChild
  else:
    ControllerClass = GeneralController
    ChildClass = GeneralChild

  params.add_hparam('controller_search_whole_channels', FLAGS.controller_search_whole_channels)

  child_model = ChildClass(
    images,
    labels,
    params=params,
  )

  if params.fixed_arc is None:
    controller_model = ControllerClass(
      params=params)

    child_model.connect_controller(controller_model)
    controller_model.build_trainer(child_model)

    controller_ops = {
      "train_step": controller_model.train_step,
      "loss": controller_model.loss,
      "train_op": controller_model.train_op,
      "lr": controller_model.lr,
      "grad_norm": controller_model.grad_norm,
      "valid_acc": controller_model.valid_acc,
      "optimizer": controller_model.optimizer,
      "baseline": controller_model.baseline,
      "entropy": controller_model.sample_entropy,
      "sample_arc": controller_model.sample_arc,
      "skip_rate": controller_model.skip_rate,
    }
  else:
    assert not FLAGS.controller_training, (
      "--child_fixed_arc is given, cannot train controller")
    child_model.connect_controller(None)
    controller_ops = None

  child_ops = {
    "global_step": child_model.global_step,
    "loss": child_model.loss,
    "train_op": child_model.train_op,
    "lr": child_model.lr,
    "grad_norm": child_model.grad_norm,
    "train_acc": child_model.train_acc,
    "optimizer": child_model.optimizer,
    "num_train_batches": child_model.num_train_batches,
  }

  ops = {
    "child": child_ops,
    "controller": controller_ops,
    "eval_every": child_model.num_train_batches * FLAGS.eval_every_epochs,
    "eval_func": child_model.eval_once,
    "num_train_batches": child_model.num_train_batches,
  }

  return ops, params

def cnn_trainer(params, images, labels):
  
  g = tf.Graph()
  with g.as_default():
    ops, params = get_ops(params, images, labels)
    child_ops = ops["child"]
    controller_ops = ops["controller"]

    controller_train_steps=50
    if params.search_for is 'macro':
      controller_train_steps = 50
    elif params.search_for is 'micro':
      controller_train_steps = 30

    saver = tf.train.Saver(max_to_keep=2)
    checkpoint_saver_hook = tf.train.CheckpointSaverHook(
      params.output_dir, save_steps=child_ops["num_train_batches"], saver=saver)

    hooks = [checkpoint_saver_hook]
    if FLAGS.child_sync_replicas:
      sync_replicas_hook = child_ops["optimizer"].make_session_run_hook(True)
      hooks.append(sync_replicas_hook)
    if FLAGS.controller_training and FLAGS.controller_sync_replicas:
      sync_replicas_hook = controller_ops["optimizer"].make_session_run_hook(True)
      hooks.append(sync_replicas_hook)

    print("-" * 80)
    print("Starting session")
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.train.SingularMonitoredSession(
      config=config, hooks=hooks, checkpoint_dir=params.output_dir) as sess:
        start_time = time.time()
        while True:
          run_ops = [
            child_ops["loss"],
            child_ops["lr"],
            child_ops["grad_norm"],
            child_ops["train_acc"],
            child_ops["train_op"],
          ]
          loss, lr, gn, tr_acc, _ = sess.run(run_ops)
          global_step = sess.run(child_ops["global_step"])

          actual_step = global_step

          epoch = actual_step // ops["num_train_batches"]
          curr_time = time.time()
          if global_step % params.log_every == 0:
            log_string = ""
            log_string += "epoch={:<6d}".format(epoch)
            log_string += "ch_step={:<6d}".format(global_step)
            log_string += " loss={:<8.6f}".format(loss)
            log_string += " lr={:<8.4f}".format(lr)
            log_string += " |g|={:<8.4f}".format(gn)
            log_string += " tr_acc={:<3d}/{:>3d}".format(
                tr_acc, params.batch_size)
            log_string += " mins={:<10.2f}".format(
                float(curr_time - start_time) / 60)
            print(log_string)
            
          if actual_step % ops["eval_every"] == 0:
            if (params.controller_training): #and
                 #epoch % FLAGS.controller_train_every == 0):

              print("Epoch {}: Training controller".format(epoch))

              controller_num_aggregate = 20
              if params.search_for == "macro":
                controller_num_aggregate = 20
              elif params.search_for == "micro":
                controller_num_aggregate = 10

              for ct_step in range(controller_train_steps *
                                    controller_num_aggregate):
                run_ops = [
                  controller_ops["loss"],
                  controller_ops["entropy"],
                  controller_ops["lr"],
                  controller_ops["grad_norm"],
                  controller_ops["valid_acc"],
                  controller_ops["baseline"],
                  controller_ops["skip_rate"],
                  controller_ops["train_op"],
                ]
                loss, entropy, lr, gn, val_acc, bl, skip, _ = sess.run(run_ops)
                controller_step = sess.run(controller_ops["train_step"])

                if ct_step % params.log_every == 0:
                  curr_time = time.time()
                  log_string = ""
                  log_string += "ctrl_step={:<6d}".format(controller_step)
                  log_string += " loss={:<7.3f}".format(loss)
                  log_string += " ent={:<5.2f}".format(entropy)
                  log_string += " lr={:<6.4f}".format(lr)
                  log_string += " |g|={:<8.4f}".format(gn)
                  log_string += " acc={:<6.4f}".format(val_acc)
                  log_string += " bl={:<5.2f}".format(bl)
                  log_string += " mins={:<.2f}".format(
                      float(curr_time - start_time) / 60)
                  print(log_string)

              print("Here are 10 architectures")
              for _ in range(10):
                arc, acc = sess.run([
                  controller_ops["sample_arc"],
                  controller_ops["valid_acc"],
                ])
                if params.search_for == "micro":
                  normal_arc, reduce_arc = arc
                  print(np.reshape(normal_arc, [-1]))
                  print(np.reshape(reduce_arc, [-1]))
                else:
                  start = 0
                  for layer_id in range(params.child_num_layers):
                    if FLAGS.controller_search_whole_channels:
                      end = start + 1 + layer_id
                    else:
                      end = start + 2 * FLAGS.child_num_branches + layer_id
                    print(np.reshape(arc[start: end], [-1]))
                    start = end
                print("val_acc={:<6.4f}".format(acc))
                print("-" * 80)

            print("Epoch {}: Eval".format(epoch))
            if params.fixed_arc is None:
              ops["eval_func"](sess, "valid")
            ops["eval_func"](sess, "test")

          if epoch >= params.num_epochs:
            break

