import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops

class UserGRUCell4Rec(tf.nn.rnn_cell.MultiRNNCell):
  """
  UserGRU cell for HGRU4Rec
  """

  def __init__(self, cells, state_is_tuple=True):
    super(UserGRUCell4Rec, self).__init__(cells, state_is_tuple=state_is_tuple)

  def call(self, inputs, state, sstart, ustart):
    """Run this multi-layer cell on inputs, starting from state."""
    cur_state_pos = 0
    cur_inp = inputs
    new_states = []
    for i, cell in enumerate(self._cells):
      with vs.variable_scope("cell_%d" % i):
        if self._state_is_tuple:
          if not nest.is_sequence(state):
            raise ValueError(
              "Expected state to be a tuple of length %d, but received: %s" %
              (len(self.state_size), state))
          cur_state = state[i]
        else:
          cur_state = array_ops.slice(state, [0, cur_state_pos],
                                      [-1, cell.state_size])
          cur_state_pos += cell.state_size
        o, h = cell(cur_inp, cur_state)
        h = tf.where(self.sstart, h, cur_inp)
        h = tf.where(self.ustart, tf.zeros(tf.shape(h)), h)

        new_states.append(h)
        cur_inp = h

    new_states = (tuple(new_states) if self._state_is_tuple else
                  array_ops.concat(new_states, 1))

    return cur_inp, new_states

