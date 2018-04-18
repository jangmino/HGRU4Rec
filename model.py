import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops
import numpy as np
import pandas as pd
from os import path
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

class HGRU4Rec:
  """

  """
  def __init__(self, sess, session_layers, user_layers, n_epochs=50, batch_size=50, learning_rate=0.001, momentum=0.0,
               decay=0.96, grad_cap=0, sigma=0, dropout_p_hidden_usr=0.3,
               dropout_p_hidden_ses=0.3, dropout_p_init=0.3, init_as_normal=False,
               reset_after_session=True, loss='top1', hidden_act='tanh', final_act=None, train_random_order=False,
               lmbd=0.0, session_key='session_id', item_key='item_id', time_key='created_at', user_key='user_id', n_sample=0,
               sample_alpha=0.75, user_propagation_mode='init',
               user_to_output=False, user_to_session_act='tanh', n_items=4, checkpoint_dir='', log_dir=''):

    self.sess = sess
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
    self.n_items = n_items
    self.log_dir = log_dir
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

    self.checkpoint_dir = checkpoint_dir
    if not path.isdir(self.checkpoint_dir):
      raise Exception("[!] Checkpoint Dir not found")

    self.build_model()
    self.sess.run(tf.global_variables_initializer())
    self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

    if self.is_training:
      return

    # # use self.predict_state to hold hidden states during prediction.
    # self.predict_state = [np.zeros([self.batch_size, self.rnn_size], dtype=np.float32) for _ in xrange(self.layers)]
    # ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
    # if ckpt and ckpt.model_checkpoint_path:
    #   self.saver.restore(sess, '{}/gru-model-{}'.format(self.checkpoint_dir, args.test_model))

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
    with tf.name_scope("top1"):
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
      #return super(HGRU4Rec.UserGRUCell4Rec, self).call(inputs, state)

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
          h = tf.where(self.hgru4rec.sstart, h, cur_state, name='sel_hu_1')
          h = tf.where(self.hgru4rec.ustart, tf.zeros(tf.shape(h)), h, name='sel_hu_2')

          new_states.append(h)
          cur_inp = h

      new_states = (tuple(new_states) if self._state_is_tuple else
                    array_ops.concat(new_states, 1))

      return cur_inp, new_states

  def build_model(self):
    """

    :return:
    """
    self.X = tf.placeholder(tf.int32, [self.batch_size], name='input_x')
    self.Y = tf.placeholder(tf.int32, [self.batch_size], name='output_y')
    self.Hs = [tf.placeholder(tf.float32, [self.batch_size, s_size], name='Hs') for s_size in
                  self.session_layers]
    self.Hu = [tf.placeholder(tf.float32, [self.batch_size, u_size], name='Hu') for u_size in
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
      embedding = tf.get_variable('embedding', [self.n_items, self.session_layers[0]], initializer=initializer)
      softmax_W = tf.get_variable('softmax_w', [self.n_items, self.session_layers[0]], initializer=initializer)
      softmax_b = tf.get_variable('softmax_b', [self.n_items], initializer=tf.constant_initializer(0.0))

      cells = []
      for s_size in self.session_layers:
        cell = tf.nn.rnn_cell.GRUCell(s_size, activation=self.hidden_act)
        cells.append(tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout_p_hidden_ses))
      stacked_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

      input_states=[]
      for j in range(len(self.session_layers)):
        h_s_init = tf.layers.dropout(tf.layers.dense(self.Hu_new[-1], self.session_layers[j]),
                                   rate=self.dropout_p_init, training=self.is_training,
                                   name='h_s_init_{}'.format(j))
        h_s = tf.where(self.sstart, h_s_init, self.Hs[j], name='sel_hs_1_{}'.format(j))
        h_s = tf.where(self.ustart, tf.zeros(tf.shape(h_s)), h_s, name='sel_hs_2_{}'.format(j))
        input_states.append(h_s)

      inputs = tf.nn.embedding_lookup(embedding, self.X, name='embedding_x')
      output, state = stacked_cell(inputs,
                                   tuple(input_states)
                                   )
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
        tf.summary.scalar('cost', self.cost)
      else:
        logits = tf.matmul(output, softmax_W, transpose_b=True) + softmax_b
        self.yhat = self.final_activation(logits)

    if not self.is_training:
      return

    with tf.name_scope("optimizer"):
      self.lr = tf.maximum(1e-5,
                         tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay,
                                                    staircase=True))
      '''
      Try different optimizers.
      '''
      tf.summary.scalar('lr', self.lr)
      optimizer = tf.train.AdamOptimizer(self.lr)

      tvars = tf.trainable_variables()
      gvs = optimizer.compute_gradients(self.cost, tvars)
      if self.grad_cap > 0:
        capped_gvs = [(tf.clip_by_norm(grad, self.grad_cap), var) for grad, var in gvs]
      else:
        capped_gvs = gvs
      self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)

    self.merged = tf.summary.merge_all()
    self.train_writer = tf.summary.FileWriter(path.join(self.log_dir, 'train'), self.sess.graph)

  def preprocess_data(self, data):
      # sort by user and time key in order
      data.sort_values([self.user_key, self.session_key, self.time_key], inplace=True)
      data.reset_index(drop=True, inplace=True)
      offset_session = np.r_[0, data.groupby([self.user_key, self.session_key], sort=False).size().cumsum()[:-1]]
      user_indptr = np.r_[0, data.groupby(self.user_key, sort=False)[self.session_key].nunique().cumsum()[:-1]]
      return user_indptr, offset_session

  def iterate(self, data, offset_sessions, user_indptr, reset_state=True, is_validation=False):
    """

    :param data:
    :param offset_sessions:
    :param user_indptr:
    :param reset_state:
    :return:
    """

    # variables to manage iterations over users
    n_users = len(user_indptr)
    offset_users = offset_sessions[user_indptr]
    user_idx_arr = np.arange(n_users - 1)
    user_iters = np.arange(self.batch_size)
    user_maxiter = user_iters.max()
    user_start = offset_users[user_idx_arr[user_iters]]
    user_end = offset_users[user_idx_arr[user_iters] + 1]

    # variables to manage iterations over sessions
    session_iters = user_indptr[user_iters]
    session_start = offset_sessions[session_iters]
    session_end = offset_sessions[session_iters + 1]

    sstart = np.zeros((self.batch_size,), dtype=np.bool)
    ustart = np.zeros((self.batch_size,), dtype=np.bool)
    finished = False
    n = 0
    c = []
    summary = None

    Hs_new = [np.zeros([self.batch_size, s_size], dtype=np.float32) for s_size in self.session_layers]
    Hu_new = [np.zeros([self.batch_size, u_size], dtype=np.float32) for u_size in self.user_layers]
    while not finished:
      session_minlen = (session_end - session_start).min()
      out_idx = data.ItemIdx.values[session_start]
      for i in range(session_minlen - 1):
        in_idx = out_idx
        out_idx = data.ItemIdx.values[session_start + i + 1]
        #if self.n_sample:
          #   sample = self.neg_sampler.next_sample()
          #   y = np.hstack([out_idx, sample])
          # else:
        y = out_idx

        feed_dict = {self.X: in_idx, self.Y: y, self.sstart: sstart, self.ustart: ustart}
        for j in range(len(self.Hs)):
          feed_dict[self.Hs[j]] = Hs_new[j]
        for j in range(len(self.Hu)):
          feed_dict[self.Hu[j]] = Hu_new[j]

        fetches = []
        if is_validation == False:
          fetches = [self.merged, self.cost, self.Hs_new, self.Hu_new, self.global_step, self.lr, self.train_op]
        else:
          fetches = [self.merged, self.cost, self.Hs_new, self.Hu_new]
        summary, cost, Hs_new, Hu_new, step, lr, _ = self.sess.run(fetches, feed_dict)

        n += 1

        if is_validation == False:
          self.train_writer.add_summary(summary, step)
        # reset sstart and ustart
        sstart = np.zeros_like(sstart, dtype=np.bool)
        ustart = np.zeros_like(ustart, dtype=np.bool)
        c.append(cost)
        if np.isnan(cost):
          logger.error('NaN error!')
          self.error_during_train = True
          return
      session_start = session_start + session_minlen - 1
      session_start_mask = np.arange(len(session_iters))[(session_end - session_start) <= 1]
      sstart[session_start_mask] = True
      for idx in session_start_mask:
        session_iters[idx] += 1
        if session_iters[idx] + 1 >= len(offset_sessions):
          finished = True
          break
        session_start[idx] = offset_sessions[session_iters[idx]]
        session_end[idx] = offset_sessions[session_iters[idx] + 1]

      # reset the User hidden state at user change
      user_change_mask = np.arange(len(user_iters))[(user_end - session_start <= 0)]
      ustart[user_change_mask] = True
      for idx in user_change_mask:
        user_maxiter += 1
        if user_maxiter + 1 >= len(offset_users):
          finished = True
          break
        user_iters[idx] = user_maxiter
        user_start[idx] = offset_users[user_maxiter]
        user_end[idx] = offset_users[user_maxiter + 1]
        session_iters[idx] = user_indptr[user_maxiter]
        session_start[idx] = offset_sessions[session_iters[idx]]
        session_end[idx] = offset_sessions[session_iters[idx] + 1]
    avgc = np.mean(c)

    return avgc

  def fit(self, train_data, valid_data=None, patience=3):
    """
    :param train_data:
    :param valid_data:
    :return:
    """
    self.error_during_train = False

    itemids = train_data[self.item_key].unique()
    self.itemidmap = pd.Series(data=np.arange(self.n_items), index=itemids)
    train_data = pd.merge(train_data,
                          pd.DataFrame({self.item_key: itemids, 'ItemIdx': self.itemidmap[itemids].values}),
                          on=self.item_key, how='inner')
    user_indptr, offset_sessions = self.preprocess_data(train_data)

    user_indptr_valid, offset_sessions_valid = None, None
    if valid_data is not None:
      valid_data = pd.merge(valid_data,
                            pd.DataFrame({self.item_key: itemids, 'ItemIdx': self.itemidmap[itemids].values}),
                            on=self.item_key, how='inner')
      user_indptr_valid, offset_sessions_valid = self.preprocess_data(valid_data)

    epoch = 0
    best_valid = None
    my_patience = patience
    while epoch < self.n_epochs and my_patience > 0:
      train_cost = self.iterate(train_data, offset_sessions, user_indptr)
      if np.isnan(train_cost):
        print('Epoch {}: Nan error!'.format(epoch, train_cost))
        return

      if valid_data is not None:
        valid_cost = self.iterate(valid_data, offset_sessions_valid, user_indptr_valid)
        if best_valid is None or valid_cost < best_valid:
          best_valid = valid_cost
        logger.info(
          'Epoch {} - train cost: {:.4f} - valid cost: {:.4f}'.format(epoch, train_cost, valid_cost, my_patience)
        )
      else:
        logger.info('Epoch {} -train cost: {:.4f}'.format(epoch, train_cost))

      epoch += 1

      #self.saver.save(self.sess, '{}/hgru-model'.format(self.checkpoint_dir), global_step=epoch)
