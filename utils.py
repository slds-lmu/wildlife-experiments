"""Utility functions for experiments."""

import itertools

import tensorflow as tf
import os
import random
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping


def product_dict(**kwargs):
    """Get cartesian product from dictionary of lists."""
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def seed_everything(seed: int) -> None:
    """At least we tried."""
    seed = int(seed)
    tf.random.set_seed(seed)
    tf.compat.v1.set_random_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    session_conf = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
    )
    sess = tf.compat.v1.Session(
        graph=tf.compat.v1.get_default_graph(), config=session_conf
    )
    tf.compat.v1.keras.backend.set_session(sess)


class MyEarlyStopping(EarlyStopping):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.baseline_attained = False

    def on_epoch_end(self, epoch, logs=None):
        if not self.baseline_attained:
            current = self.get_monitor_value(logs)
            if current is None:
                return

            if self.monitor_op(current, self.baseline):
                if self.verbose > 0:
                    print('Baseline attained.')
                self.baseline_attained = True
            else:
                return

        super(MyEarlyStopping, self).on_epoch_end(epoch, logs)