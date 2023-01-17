"""Utility functions for experiments."""

import itertools


def product_dict(**kwargs):
    """Get cartesian product from dictionary of lists."""
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))