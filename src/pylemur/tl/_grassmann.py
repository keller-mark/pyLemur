import numpy as np
import dask.array as da


def grassmann_map(x, base_point, use_dask=True):
    if use_dask:
        if base_point.shape[0] == 0 or base_point.shape[1] == 0:
            return base_point
        elif da.isnan(x).any():
            # Return an object with the same shape as x filled with nan
            return da.full(x.shape, da.nan)
        else:
            full_matrices = False
            u, s, vt = da.linalg.svd(x)
            if not full_matrices:
                m, n = x.shape
                u = u[:, :n]
                vt = vt[:m, :]

            return (base_point @ vt.T) @ da.diag(da.cos(s)) @ vt + u @ da.diag(da.sin(s)) @ vt
    else:
        if base_point.shape[0] == 0 or base_point.shape[1] == 0:
            return base_point
        elif np.isnan(x).any():
            # Return an object with the same shape as x filled with nan
            return np.full(x.shape, np.nan)
        else:
            full_matrices = False
            u, s, vt = np.linalg.svd(x, full_matrices=full_matrices)

            return (base_point @ vt.T) @ np.diag(np.cos(s)) @ vt + u @ np.diag(np.sin(s)) @ vt
        


def grassmann_log(p, q):
    n = p.shape[0]
    k = p.shape[1]

    if n == 0 or k == 0:
        return p
    else:
        z = q.T @ p
        At = q.T - z @ p.T
        # Translate `lm.fit(z, At)$coefficients` to python
        Bt = da.linalg.lstsq(z, At)[0]
        full_matrices = True # numpy param
        u, s, vt = da.linalg.svd(Bt.T)
        u = u[:, :k]
        s = s[:k]
        vt = vt[:k, :]
        return u @ da.diag(da.arctan(s)) @ vt


def grassmann_project(x):
    return da.linalg.qr(x)[0]


def grassmann_project_tangent(x, base_point):
    return x - base_point @ base_point.T @ x


def grassmann_random_point(n, k):
    x = np.random.randn(n, k)
    return grassmann_project(x)


def grassmann_random_tangent(base_point):
    x = np.random.randn(*base_point.shape)
    return grassmann_project_tangent(x, base_point)


def grassmann_angle_from_tangent(x, normalized=True):
    full_matrices = True # numpy param
    compute_uv = False # numpy param
    _u, thetas, _v = da.linalg.svd(x) / np.pi * 180
    if normalized:
        return np.minimum(thetas, 180 - thetas).max()
    else:
        return thetas[0]


def grassmann_angle_from_point(x, y):
    return grassmann_angle_from_tangent(grassmann_log(y, x))
