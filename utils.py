"""Utility functions for experiments."""

import itertools
import tensorflow as tf
import os
import random
import numpy as np


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
