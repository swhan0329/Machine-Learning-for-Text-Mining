import numpy as np

# for realsim dataset
# lambda_ = 7230.875

# for covtype dataset
# lambda_ = 3631.3203125

# please refer wikipedea
# https://en.wikipedia.org/wiki/Conjugate_gradient_method
# this function solve  Hd = -g
def conjugateGradient(X, I, grad, lambda_, tol=1e-1, max_iter=100):
    """conjugateGradient

    :param X: shape = (N, M)
    :param I: can be a binary vector, shape = (N,),
              or a list of indices as defined in the handout.
    :param grad: shape = (M,)
    :param lambda_:
    :param tol:
    :param max_iter:
    """
    # Hessian vector product
    def Hv(X, I, v):
        ret = v + (2.0 * lambda_ / X.shape[0]) * X[I].transpose().dot(X[I].dot(v))
        return ret

    # initial
    d = np.random.rand(X.shape[1])

    Hd = Hv(X, I, d)
    r = -grad - Hd
    p = r
    rsold = r.T.dot(r)

    for cg_iter in range(max_iter):
        Ap = Hv(X, I, p)
        alpha = rsold / p.T.dot(Ap)
        d = d + alpha * p
        r = r - alpha * Ap
        rsnew = r.T.dot(r)
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return d, cg_iter + 1
