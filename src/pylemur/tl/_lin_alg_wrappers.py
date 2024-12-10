from typing import NamedTuple

import numpy as np
import dask.array as da
from scipy.sparse import issparse

import sklearn.decomposition as skd
import dask_ml.decomposition as dmd


class PCA(NamedTuple):
    embedding: np.ndarray
    coord_system: np.ndarray
    offset: np.ndarray


def fit_pca(Y, n, center=True):
    """
    Calculate the PCA of a given data matrix Y.

    Parameters
    ----------
    Y : array-like, shape (n_samples, n_features)
        The input data matrix.
    n : int
        The number of principal components to return.
    center : bool, default=True
        If True, the data will be centered before computing the covariance matrix.

    Returns
    -------
    pca : sklearn.decomposition.PCA
        The PCA object.
    """
    if center:
        pca = dmd.PCA(n_components=n)
        emb = pca.fit_transform(Y)
        coord_system = pca.components_
        mean = pca.mean_
    else:
        n_components = n
        u, s, v = da.linalg.svd_compressed(
            Y, n_components
        )
        components_ = v
        emb = u * s
        coord_system = components_
        mean = da.zeros(Y.shape[1])
    return PCA(emb, coord_system, mean)


def ridge_regression(Y, X, ridge_penalty=0, weights=None):
    """
    Calculate the ridge regression of a given data matrix Y.

    Parameters
    ----------
    Y : array-like, shape (n_samples, n_features)
        The input data matrix.
    X : array-like, shape (n_samples, n_coef)
        The input data matrix.
    ridge_penalty : float, default=0
        The ridge penalty.
    weights : array-like, shape (n_features,)
        The weights to apply to each feature.

    Returns
    -------
    ridge: array-like, shape (n_coef, n_features)
    """
    n_coef = X.shape[1]
    n_samples = X.shape[0]
    n_feat = Y.shape[1]
    assert Y.shape[0] == n_samples
    if weights is None:
        weights = da.ones(n_samples)
    assert len(weights) == n_samples

    if isinstance(weights, list):
        weights = da.from_array(weights)

    if np.ndim(ridge_penalty) == 0 or len(ridge_penalty) == 1:
        ridge_penalty = da.eye(n_coef) * ridge_penalty
    elif np.ndim(ridge_penalty) == 1:
        assert len(ridge_penalty) == n_coef
        ridge_penalty = da.diag(ridge_penalty)
    elif np.ndim(ridge_penalty) == 1:
        assert ridge_penalty.shape == (n_coef, n_coef)
        pass
    else:
        raise ValueError("ridge_penalty must be a scalar, 1d array, or 2d array")

    ridge_penalty_sq = da.sqrt(da.sum(weights)) * (ridge_penalty.T @ ridge_penalty)
    weights_sqrt = da.sqrt(weights)
    X_ext = da.vstack([multiply_along_axis(X, weights_sqrt, 0), ridge_penalty_sq])
    Y_ext = da.vstack([multiply_along_axis(Y, weights_sqrt, 0), da.zeros((n_coef, n_feat))])

    ridge = da.linalg.lstsq(X_ext, Y_ext)[0]
    return ridge


def multiply_along_axis(A, B, axis):
    # Copied from https://stackoverflow.com/a/71750176/604854
    try:
        A = A.compute()
    except AttributeError:
        # Was not a dask array
        pass
    try:
        B = B.compute()
    except AttributeError:
        # Was not a dask array
        pass
    if issparse(A):
        A = A.todense()
    if issparse(B):
        B = B.todense()
    try:
        result = mult_along_axis(A, B, axis)
    except ValueError:
        result = mult_along_axis_alt(A, B, axis)
    return da.from_array(result)

def mult_along_axis(A, B, axis):
    # ensure we're working with Numpy arrays
    A = np.array(A)
    B = np.array(B)

    # shape check
    if axis >= A.ndim:
        raise ValueError(f"AxisError({axis}, {A.ndim}")
    if A.shape[axis] != B.size:
        raise ValueError(
            "Length of 'A' along the given axis must be the same as B.size"
        )

    # np.broadcast_to puts the new axis as the last axis, so 
    # we swap the given axis with the last one, to determine the
    # corresponding array shape. np.swapaxes only returns a view
    # of the supplied array, so no data is copied unnecessarily.
    shape = np.swapaxes(A, A.ndim-1, axis).shape

    # Broadcast to an array with the shape as above. Again, 
    # no data is copied, we only get a new look at the existing data.
    B_brc = np.broadcast_to(B, shape)

    # Swap back the axes. As before, this only changes our "point of view".
    B_brc = np.swapaxes(B_brc, A.ndim-1, axis)

    return A * B_brc

def mult_along_axis_alt(A, B, axis):
    # Copied from https://stackoverflow.com/a/71750176/604854
    return np.swapaxes(np.swapaxes(A, axis, -1) * B, -1, axis)
