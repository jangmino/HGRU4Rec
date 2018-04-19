# HGRU4Rec for TensorFlow

A TensorFlow implementation of HGRU4Rec, which is based on the original theano version https://github.com/mquad/hgru4rec.
Also, it refers an implementation of GRU4Rec by https://github.com/Songweiping/GRU4Rec_TensorFlow.

Note that I just implemented and verified only training process. 

# Requirements

Python: 3.5

TensorFlow: 1.7

Pandas >= 0.22

tables>= 3.4.2

# Main points
Please refer inner class `class UserGRUCell4Rec(tf.nn.rnn_cell.MultiRNNCell)`:

- It is subclassing `MultiRNNCell` which is for GRU_user. You can find my main TF conversion in the overriding `call()`.

A difference with the author's implementation:
- Within `tf.variable_scope('session_gru')`, I extend the initialization mechanism for `input_states` for whole layered cells.
- Original version does the initialization on the first layer onley.

# Note

Currently XING data used in the original paper is not available. Therefore, I made a verifying data set from https://www.kaggle.com/retailrocket/ecommerce-dataset.

You can refer `build_dataset.py` from `data/xing` under https://github.com/mquad/hgru4rec to make your dataset.
