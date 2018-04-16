from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import model

flags = tf.app.flags

FLAGS = flags.FLAGS


class HGRU4RecTest(tf.test.TestCase):

  def setUp(self):
    print('setup is called')


  def testCustomCell(self):
    cell = tf.nn.rnn_cell.GRUCell(10, 10)
    m = model.UserGRUCell4Rec([cell] * 5)
    return True


if __name__ == "__main__":
  tf.test.main()
