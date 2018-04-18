import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
from os import path
from datetime import datetime as dt

import model

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

parser = argparse.ArgumentParser()
parser.add_argument('session_layers', type=str)
parser.add_argument('user_layers', type=str)
parser.add_argument('--loss', type=str, default='top1')
parser.add_argument('--hidden_act', type=str, default='tanh')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--momentum', type=float, default=0.0)
parser.add_argument('--dropout_p_hidden_usr', type=float, default=0.3)
parser.add_argument('--dropout_p_hidden_ses', type=float, default=0.3)
parser.add_argument('--dropout_p_init', type=float, default=0.3)
parser.add_argument('--decay', type=float, default=0.96)
parser.add_argument('--grad_cap', type=float, default=0.0)
parser.add_argument('--sigma', type=float, default=0.0)
parser.add_argument('--n_epochs', type=int, default=50)
parser.add_argument('--init_as_normal', type=int, default=0)
parser.add_argument('--reset_after_session', type=int, default=1)
parser.add_argument('--user_key', type=str, default='visitorid')
parser.add_argument('--item_key', type=str, default='itemid')
parser.add_argument('--session_key', type=str, default='session_id')
parser.add_argument('--time_key', type=str, default='timestamp')
parser.add_argument('--save_to', type=str, default=None)
parser.add_argument('--load_from', type=str, default=None)
parser.add_argument('--early_stopping', action='store_true', default=False)
parser.add_argument('--hdf_path', type=str, default='')
parser.add_argument('--checkpoint_dir', type=str, default=r'./model')
parser.add_argument('--log_dir', type=str, default=r'./log')
# user bias parameters
parser.add_argument('--user_to_ses_act', type=str, default='tanh')
parser.add_argument('--user_propagation_mode', type=str, default='all')
parser.add_argument('--user_to_output', type=int, default=1)
args = parser.parse_args()

sessions_path = args.hdf_path
logger.info('Loading data from: {}'.format(sessions_path))
train_data = pd.read_hdf(sessions_path, 'train')
test_data = pd.read_hdf(sessions_path, 'valid_train') if args.early_stopping else None

session_layers = [int(x) for x in args.session_layers.split(',')]
user_layers = [int(x) for x in args.user_layers.split(',')]

logger.info('session_layers: {}'.format(args.session_layers))
logger.info('user_layers: {}'.format(args.user_layers))
logger.info('loss: {}'.format(args.loss))
logger.info('hidden_act: {}'.format(args.hidden_act))
logger.info('batch_size: {}'.format(args.batch_size))
logger.info('dropout_p_hidden_usr: {}'.format(args.dropout_p_hidden_usr))
logger.info('dropout_p_hidden_ses: {}'.format(args.dropout_p_hidden_ses))
logger.info('dropout_p_init: {}'.format(args.dropout_p_init))
logger.info('init_as_normal: {}'.format(args.init_as_normal))
logger.info('grad_cap: {}'.format(args.grad_cap))
logger.info('sigma: {}'.format(args.sigma))
logger.info('decay (only for rmsprop): {}'.format(args.decay))
logger.info('')
logger.info('TRAINING:')
logger.info('learning_rate: {}'.format(args.learning_rate))
logger.info('n_epochs: {}'.format(args.n_epochs))
logger.info('reset_after_session: {}'.format(args.reset_after_session))
logger.info('n_epochs: {}'.format(args.n_epochs))
logger.info('early_stopping: {}'.format(args.early_stopping))
logger.info('')

itemids = train_data[args.item_key].unique()
n_items = len(itemids)

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
with tf.Session(config=gpu_config) as sess:
  m = model.HGRU4Rec(sess,
        session_layers=session_layers,
        user_layers=user_layers,
        loss=args.loss,
        hidden_act=args.hidden_act,
        dropout_p_hidden_usr=args.dropout_p_hidden_usr,
        dropout_p_hidden_ses=args.dropout_p_hidden_ses,
        dropout_p_init=args.dropout_p_init,
        decay=args.decay,
        grad_cap=args.grad_cap,
        sigma=args.sigma,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        init_as_normal=bool(args.init_as_normal),
        reset_after_session=bool(args.reset_after_session),
        n_epochs=args.n_epochs,
        user_key=args.user_key,
        session_key=args.session_key,
        item_key=args.item_key,
        time_key=args.time_key,
        user_to_session_act=args.user_to_ses_act,
        user_propagation_mode=args.user_propagation_mode,
        user_to_output=bool(args.user_to_output),
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        n_items=n_items )

  t0 = dt.now()
  logger.info('Training started')
  m.fit(train_data,
            valid_data=test_data if args.early_stopping else None,
            patience=3,
            margin=1.003
        )
  logger.info('Training completed in {}'.format(dt.now() - t0))


