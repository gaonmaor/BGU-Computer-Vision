import numpy as np
import collections


def obj_keys(obj):
    return (k for k in obj.keys() if not k.startswith('__'))


def y2row(y, size=8):
    return np.asarray([(int(c) * 2 - 1) for c in np.binary_repr(y, size)])
