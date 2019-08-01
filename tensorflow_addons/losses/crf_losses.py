from tensorflow.python.keras.losses import sparse_categorical_crossentropy, \
    categorical_crossentropy

import tensorflow as tf
from tensorflow_addons.utils import keras_utils


def crf_nll(y_true, y_pred):
    crf, idx = y_pred._keras_history[:2]

    node = crf._inbound_nodes[idx]

    nloglik = crf.get_negative_log_likelihood(y_true)

    return nloglik


def crf_loss(y_true, y_pred):
    # TODO: change to tf 2.0 class based implementation
    crf, idx = y_pred._keras_history[:2]

    if crf.learn_mode == 'join':
        return crf_nll(y_true, y_pred)
    else:
        if crf.sparse_target:
            return sparse_categorical_crossentropy(y_true, y_pred)
        else:
            return categorical_crossentropy(y_true, y_pred)


@keras_utils.register_keras_custom_object
class CrfLoss(tf.keras.losses.Loss):
    def __init__(self,
                 reduction=tf.keras.losses.Reduction.NONE,
                 name='crf_loss'):
        super(CrfLoss, self).__init__(name=name, reduction=reduction)

    def call(self, y_true, y_pred):
        return crf_loss(y_true, y_pred)

    def get_config(self):
        config = {}
        base_config = super(CrfLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
