from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pandas as pd
from os import path

import model

flags = tf.app.flags

FLAGS = flags.FLAGS


class HGRU4RecTest(tf.test.TestCase):

  def setUp(self):
    print('setup is called')


  # def testCustomCell(self):
  #   cell = tf.nn.rnn_cell.GRUCell(10, 10)
  #   m = model.UserGRUCell4Rec([cell] * 5)
  #   return True

  def testBuildModel(self):
    train_data = pd.read_hdf(r'.\data\sessions.hdf', 'whole_data')
    itemids = train_data['item_id'].unique()
    n_items = len(itemids)
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config=gpu_config) as sess:
      hgru4rec = model.HGRU4Rec(sess, [50], [50], batch_size=2, n_items=n_items, n_epochs=10,
                                checkpoint_dir=r'.\model',
                                log_dir=r'.\log'
                                )
      hgru4rec.fit(train_data)

  # def testBuildModel(self):
  #   # train_data = pd.read_hdf(r'c:\Users\wmp\theano\hgru4rec\data\xing\dense\last-session-out\sessions.hdf', 'whole_data')
  #   # itemids = train_data['item_id'].unique()
  #   # n_items = len(itemids)
  #   gpu_config = tf.ConfigProto()
  #   gpu_config.gpu_options.allow_growth = True
  #   with tf.Session(config=gpu_config) as sess:
  #     hgru4rec = model.HGRU4Rec(sess, [5, 5], [5, 5], batch_size=2, n_items=4,
  #                               checkpoint_dir=r'/Users/jangmino/tensorflow/HGRU4Rec/model',
  #                               log_dir=r'/Users/jangmino/tensorflow/HGRU4Rec/log'
  #                               )
  #     # hgru4rec.fit(train_data)


if __name__ == "__main__":
  tf.test.main()
