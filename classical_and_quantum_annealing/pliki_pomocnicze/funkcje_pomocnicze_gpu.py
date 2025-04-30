import cupy as cp

def calculate_energy_gpu(J: cp.ndarray, h: cp.ndarray, state: cp.ndarray):
    # Zakładamy, że J jest hermitowska z czynnikiem 1/2
    n, _ = J.shape
    A = cp.multiply(-1/2, J)
    B = cp.matmul(A, state) - h.reshape(n, 1)
    C = cp.multiply(state, B)
    return cp.sum(C, axis=0)