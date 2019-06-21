# -*- coding: utf-8 -*-
import tensorflow as tf

tf.app.flags.DEFINE_float("learning_rate", 1e-4, "initial learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 1.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 50, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("hidden_size", 512, "Size of each model layer.")
tf.app.flags.DEFINE_integer("emb_size", 512, "Size of character embedding.")

tf.app.flags.DEFINE_integer("steps_per_checkpoint", 50000, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("steps_per_validate",  10000, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("steps_per_train_log", 10000, "How many training steps to do per checkpoint.")

tf.app.flags.DEFINE_integer("steps_per_sample", 500, "How many training steps to sample and show the answers.")
tf.app.flags.DEFINE_integer("sample_num", 1, "How many samples to show")

tf.app.flags.DEFINE_string("model_dir", "model", "Training directory to save the model parameters.")
tf.app.flags.DEFINE_string("data_dir", "data/next_topic", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "train", "Data directory")
tf.app.flags.DEFINE_string("dev_dir", "dev", "Data directory")
tf.app.flags.DEFINE_string("device", '/gpu:0', "Device to use")

tf.app.flags.DEFINE_integer("num_topic", 10, "Number of topics in the model.")
FLAGS = tf.app.flags.FLAGS
