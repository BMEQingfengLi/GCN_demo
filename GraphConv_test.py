import numpy as np

def symmetric_normalized_laplacian(adjacency_mtx):
    degree = np.sum(adjacency_mtx, axis=1)
    D = np.diag(degree)

    D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))

    identity = np.eye(adjacency_mtx.shape[0])
    L_sym = identity - D_inv_sqrt @ adjacency_mtx @ D_inv_sqrt

    return L_sym

A = np.array([[0, 0.3, 0.7, 0.6], [0.3, 0, 0.1, 0.5], [0.7, 0.1, 0, 0.2], [0.6, 0.5, 0.2, 0]])
print(symmetric_normalized_laplacian(A))