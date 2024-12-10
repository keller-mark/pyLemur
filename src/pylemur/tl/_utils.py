import numpy as np
import dask.array as da
from scipy.sparse import issparse

def ensure_numpy(x):
    if isinstance(x, da.Array):
        return np.array(x.compute())
    return np.array(x)

def ensure_dask(x):
    if not isinstance(x, da.Array):
        return da.from_array(x)
    return x

def ensure_dense(x):
    if issparse(x):
        return x.todense()
    return x