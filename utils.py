"""Utility functions for experiments."""

import itertools

import tensorflow as tf
import os
import random
import numpy as np
from tensorflow.keras.callbacks import Callback


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


# https://stackoverflow.com/questions/53500047/stop-training-in-keras-when-accuracy-is-already-1-0

# class MyEarlyStopping(EarlyStopping):
#     """ES such that patience epochs only count when baseline is met."""
#     def __init__(self, start_from_epoch: int = 0, *args, **kw):
#         super().__init__(*args, **kw)
#         self.start_from_epoch = start_from_epoch
#
#     def on_epoch_end(self, epoch, logs=None):
#         current = self.get_monitor_value(logs)
#         if current is None or epoch < self.start_from_epoch:
#             # If no monitor value exists or still in initial warm-up stage.
#             return
#         if self.restore_best_weights and self.best_weights is None:
#             # Restore the weights after first epoch if no progress is ever made.
#             self.best_weights = self.model.get_weights()
#
#         self.wait += 1
#         if self._is_improvement(current, self.best):
#             self.best = current
#             self.best_epoch = epoch
#             if self.restore_best_weights:
#                 self.best_weights = self.model.get_weights()
#             # Only restart wait if we beat both the baseline and our previous
#             # best.
#             if self.baseline is None or self._is_improvement(
#                     current, self.baseline
#             ):
#                 self.wait = 0
#
#         # Only check after the first epoch.
#         if self.wait >= self.patience and epoch > 0:
#             self.stopped_epoch = epoch
#             self.model.stop_training = True
#             if self.restore_best_weights and self.best_weights is not None:
#                 self.model.set_weights(self.best_weights)
#
#         # self.baseline_attained = False
#
#     # def on_epoch_end(self, epoch, logs=None):
#     #     if not self.baseline_attained:
#     #         current = self.get_monitor_value(logs)
#     #         if current is None:
#     #             return
#     #
#     #         if self.monitor_op(current, self.baseline):
#     #             if self.verbose > 0:
#     #                 print('Baseline attained.')
#     #             self.baseline_attained = True
#     #         else:
#     #             return
#
#         super(MyEarlyStopping, self).on_epoch_end(epoch, logs)

# https://github.com/keras-team/keras/blob/v2.11.0/keras/callbacks.py#L1943
# Problem: loss starts low, picks up briefly, then starts decreasing --> if it doesn't
# move below the initial value in <patience> epochs, training stops (despite steady
# improvement after initial zig-zagging). Newer TF versions have option to set number of
# burn-in epochs and monitor improvement only after these, but not available here.


class MyEarlyStopping(Callback):
    """EarlyStopping callback from newest TF version."""

    def __init__(
        self,
        monitor="val_loss",
        min_delta=0,
        patience=0,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
        start_from_epoch=0,
    ):
        super().__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.start_from_epoch = start_from_epoch
        self.best = 0.
        self.best_epoch = 0

        if mode not in ["auto", "min", "max"]:
            mode = "auto"

        if mode == "min":
            self.monitor_op = np.less
        elif mode == "max":
            self.monitor_op = np.greater
        else:
            if (
                self.monitor.endswith("acc")
                or self.monitor.endswith("accuracy")
                or self.monitor.endswith("auc")
            ):
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_weights = None
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None or epoch < self.start_from_epoch:
            # If no monitor value exists or still in initial warm-up stage.
            return
        if self.restore_best_weights and self.best_weights is None:
            # Restore the weights after first epoch if no progress is ever made.
            self.best_weights = self.model.get_weights()

        self.wait += 1
        if self._is_improvement(current, self.best):
            self.best = current
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            # Only restart wait if we beat both the baseline and our previous
            # best.
            if self.baseline is None or self._is_improvement(
                current, self.baseline
            ):
                self.wait = 0

        # Only check after the first epoch.
        if self.wait >= self.patience and epoch > 0:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            if self.restore_best_weights and self.best_weights is not None:
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(
                f"Epoch {self.stopped_epoch + 1}: early stopping"
            )

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            print(
                "Early stopping conditioned on metric `%s` "
                "which is not available. Available metrics are: %s",
                self.monitor,
                ",".join(list(logs.keys())),
            )
        return monitor_value

    def _is_improvement(self, monitor_value, reference_value):
        return self.monitor_op(monitor_value - self.min_delta, reference_value)
