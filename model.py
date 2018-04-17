import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops
import numpy as np

class UserGRUCell4Recxx(tf.nn.rnn_cell.MultiRNNCell):
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
        h = tf.where(self.sstart, h, cur_state)
        h = tf.where(self.ustart, tf.zeros(tf.shape(h)), h)

        new_states.append(h)
        cur_inp = h

    new_states = (tuple(new_states) if self._state_is_tuple else
                  array_ops.concat(new_states, 1))

    return cur_inp, new_states

class HGRU4Rec:
  """

  """
  def __init__(self, session_layers, user_layers, n_epochs=10, batch_size=50, learning_rate=0.05, momentum=0.0,
               decay=0.9, grad_cap=0, sigma=0, dropout_p_hidden_usr=0.3,
               dropout_p_hidden_ses=0.0, dropout_p_init=0.0, init_as_normal=False,
               reset_after_session=True, loss='top1', hidden_act='tanh', final_act=None, train_random_order=False,
               lmbd=0.0, session_key='SessionId', item_key='ItemId', time_key='Time', user_key='UserId', n_sample=0,
               sample_alpha=0.75, user_propagation_mode='init',
               user_to_output=False, user_to_session_act='tanh'):
    self.session_layers = session_layers
    self.user_layers = user_layers
    self.n_epochs = n_epochs
    self.batch_size = batch_size
    self.dropout_p_hidden_usr = dropout_p_hidden_usr
    self.dropout_p_hidden_ses = dropout_p_hidden_ses
    self.dropout_p_init = dropout_p_init
    self.learning_rate = learning_rate
    self.decay = decay
    self.momentum = momentum
    self.sigma = sigma
    self.init_as_normal = init_as_normal
    self.reset_after_session = reset_after_session
    self.session_key = session_key
    self.item_key = item_key
    self.time_key = time_key
    self.user_key = user_key
    self.grad_cap = grad_cap
    self.train_random_order = train_random_order
    self.lmbd = lmbd

    # custom start
    self.is_training = True
    self.decay_steps = 1e4
    # custom end

    self.user_propagation_mode = user_propagation_mode
    self.user_to_output = user_to_output

    if hidden_act == 'tanh':
      self.hidden_act = self.tanh
    elif hidden_act == 'relu':
      self.hidden_act = self.relu
    else:
      raise NotImplementedError

    if loss == 'top1':
      if final_act == 'linear':
        self.final_activation = self.linear
      elif final_act == 'relu':
        self.final_activation = self.relu
      else:
        self.final_activation = self.tanh
      self.loss_function = self.top1
    else:
      raise NotImplementedError('loss {} not implemented'.format(loss))

    if hidden_act == 'relu':
      self.hidden_activation = self.relu
    elif hidden_act == 'tanh':
      self.hidden_activation = self.tanh
    else:
      raise NotImplementedError('hidden activation {} not implemented'.format(hidden_act))

    if user_to_session_act == 'relu':
      self.s_init_act = self.relu
    elif user_to_session_act == 'tanh':
      self.s_init_act = self.tanh
    else:
      raise NotImplementedError('user-to-session activation {} not implemented'.format(hidden_act))

    self.n_sample = n_sample
    self.sample_alpha = sample_alpha

  ########################ACTIVATION FUNCTIONS#########################
  def linear(self, X):
      return X
  def tanh(self, X):
      return tf.nn.tanh(X)
  def softmax(self, X):
      return tf.nn.softmax(X)
  def softmaxth(self, X):
      return tf.nn.softmax(tf.tanh(X))
  def relu(self, X):
      return tf.nn.relu(X)
  def sigmoid(self, X):
      return tf.nn.sigmoid(X)

  ############################LOSS FUNCTIONS######################
  def top1(self, yhat):
      yhatT = tf.transpose(yhat)
      term1 = tf.reduce_mean(tf.nn.sigmoid(-tf.diag_part(yhat)+yhatT)+tf.nn.sigmoid(yhatT**2), axis=0)
      term2 = tf.nn.sigmoid(tf.diag_part(yhat)**2) / self.batch_size
      return tf.reduce_mean(term1 - term2)

  class UserGRUCell4Rec(tf.nn.rnn_cell.MultiRNNCell):
    """
    UserGRU cell for HGRU4Rec
    """

    def __init__(self, cells, state_is_tuple=True, hgru4rec=None):
      super(HGRU4Rec.UserGRUCell4Rec, self).__init__(cells, state_is_tuple=state_is_tuple)
      #super().__init__(cells, state_is_tuple=state_is_tuple)
      self.hgru4rec = hgru4rec

    def call(self, inputs, state):
      """Run this multi-layer cell on inputs, starting from state."""
      return super(HGRU4Rec.UserGRUCell4Rec, self).call(inputs, state)

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
          h = tf.where(self.hgru4rec.sstart, h, cur_state)
          h = tf.where(self.hgru4rec.ustart, tf.zeros(tf.shape(h)), h)

          new_states.append(h)
          cur_inp = h

      new_states = (tuple(new_states) if self._state_is_tuple else
                    array_ops.concat(new_states, 1))

      return cur_inp, new_states

  def build_model(self):
    """

    :return:
    """
    self.X = tf.placeholder(tf.int32, [self.batch_size], name='input')
    self.Y = tf.placeholder(tf.int32, [self.batch_size], name='output')
    self.Hs = [tf.placeholder(tf.float32, [self.batch_size, s_size], name='session_state') for s_size in
                  self.session_layers]
    self.Hu = [tf.placeholder(tf.float32, [self.batch_size, u_size], name='user_state') for u_size in
                  self.user_layers]
    self.sstart = tf.placeholder(tf.bool, [self.batch_size], name='sstart')
    self.ustart = tf.placeholder(tf.bool, [self.batch_size], name='usstart')

    self.global_step = tf.Variable(0, name='global_step', trainable=False)

    # USER GRU
    with tf.variable_scope('user_gru'):
      cells = []
      for u_size in self.user_layers:
        cell = tf.nn.rnn_cell.GRUCell(u_size, activation=self.hidden_act)
        cells.append(tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout_p_hidden_usr))
      stacked_cell = self.UserGRUCell4Rec(cells, hgru4rec=self)
      output, state = stacked_cell(self.Hs[-1], tuple(self.Hu))
      self.Hu_new = state

    # SESSION GRU
    with tf.variable_scope('session_gru'):
      sigma = self.sigma if self.sigma != 0 else np.sqrt(6.0 / (self.n_items + sum(self.session_layers)))
      if self.init_as_normal:
        initializer = tf.random_normal_initializer(mean=0, stddev=sigma)
      else:
        initializer = tf.random_uniform_initializer(minval=-sigma, maxval=sigma)
      embedding = tf.get_variable('embedding', [self.n_items, self.rnn_size], initializer=initializer)
      softmax_W = tf.get_variable('softmax_w', [self.n_items, self.rnn_size], initializer=initializer)
      softmax_b = tf.get_variable('softmax_b', [self.n_items], initializer=tf.constant_initializer(0.0))

      cells = []
      for s_size in self.session_layers:
        cell = tf.nn.rnn_cell.GRUCell(s_size, activation=self.hidden_act)
        cells.append(tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout_p_hidden_ses))
      stacked_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

      h_s_init = tf.layers.dropout(tf.layers.dense(self.Hu_new[-1], self.session_layers[0]),
                                   rate=self.dropout_p_init, training=self.is_training)
      h_s = tf.where(self.sstart, h_s_init, self.Hs[0])
      self.Hs[0] = tf.where(self.ustart, tf.zeros(tf.shape(h_s)), h_s) # 이거 안될 거 같은데..

      inputs = tf.nn.embedding_lookup(embedding, self.X)
      output, state = stacked_cell(inputs, tuple(self.Hs))
      self.Hs_new = state

    if self.is_training:
      '''
      Use other examples of the minibatch as negative samples.
      '''
      sampled_W = tf.nn.embedding_lookup(softmax_W, self.Y)
      sampled_b = tf.nn.embedding_lookup(softmax_b, self.Y)
      logits = tf.matmul(output, sampled_W, transpose_b=True) + sampled_b
      self.yhat = self.final_activation(logits)
      self.cost = self.loss_function(self.yhat)
    else:
      logits = tf.matmul(output, softmax_W, transpose_b=True) + softmax_b
      self.yhat = self.final_activation(logits)

    if not self.is_training:
      return

    self.lr = tf.maximum(1e-5,
                         tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay,
                                                    staircase=True))

    '''
    Try different optimizers.
    '''
    optimizer = tf.train.AdamOptimizer(self.lr)

    tvars = tf.trainable_variables()
    gvs = optimizer.compute_gradients(self.cost, tvars)
    if self.grad_cap > 0:
      capped_gvs = [(tf.clip_by_norm(grad, self.grad_cap), var) for grad, var in gvs]
    else:
      capped_gvs = gvs
    self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)
