# funkcje pomocniczne

import os

import pandas as pd
import numpy as np


from dimod import BinaryQuadraticModel
from  collections import namedtuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

instance = namedtuple("instance", ["path", "best_energy", "name"])
small_pegasus = instance(os.path.join(ROOT, "instancje", "Pegasus", "P2_CBFM-P.txt"), -39.0, "P2")
test_pegasus = instance(os.path.join(ROOT, "instancje", "Pegasus", "P4_CBFM-P.txt"), -469.0, "P4")  # E = -469.0
full_pegasus = instance(os.path.join(ROOT, "instancje", "Pegasus", "P16_CBFM-P.txt"), -12772.0, "P16")  # E = -12772.0
small_grid = instance(os.path.join(ROOT, "instancje", "Grid", "Grid5_random.txt"), -22.75, "Grid 5")


P2 = instance(os.path.join(ROOT, "instancje", "Pegasus", "P2_CBFM-P.txt"), -39.0, "P2")
P4 = instance(os.path.join(ROOT, "instancje", "Pegasus", "P4_CBFM-P.txt"), -469.0, "P4")
P6 = instance(os.path.join(ROOT, "instancje", "Pegasus", "P6_CBFM-P.txt"), -1384.0, "P6")
P8 = instance(os.path.join(ROOT, "instancje", "Pegasus", "P8_CBFM-P.txt"), -2752.0, "P8")
P12 = instance(os.path.join(ROOT, "instancje", "Pegasus", "P12_CBFM-P.txt"), -6831.0, "P12")
P16 = instance(os.path.join(ROOT, "instancje", "Pegasus", "P16_CBFM-P.txt"), -12772.0, "P16")

Grid5 = instance(os.path.join(ROOT, "instancje", "Grid", "Grid5_random.txt"), -22.75, "Grid 5")
Grid10 = instance(os.path.join(ROOT, "instancje", "Grid", "Grid10_random.txt"), -97.25, "Grid 10")
Grid20 = instance(os.path.join(ROOT, "instancje", "Grid", "Grid20_random.txt"), -418.25, "Grid 20")
Grid50 = instance(os.path.join(ROOT, "instancje", "Grid", "Grid50_random.txt"), -2639.5, "Grid 50")
Grid100 = instance(os.path.join(ROOT, "instancje", "Grid", "Grid100_random.txt"), -10548.25, "Grid 100")

K8 = instance(os.path.join(ROOT, "instancje", "Complete", "K8_random.txt"), -9.5, "K8")


def read_instance(path: os.PathLike, convention: str = "minus_half"):
    df = pd.read_csv(path, sep=" ", header=None, comment="#", names=["i", "j", "value"])

    n = max(df[["i", "j"]].max())
    h = np.zeros(n)
    J = np.zeros((n, n))
    
    for row in df.itertuples():
        if row.i == row.j:
            h[row.i - 1] = row.value
        elif row.i > row.j:
            J[row.j - 1, row.i - 1] = row.value  # by zachować górnotrójkątność
        else:
            J[row.i - 1, row.j - 1] = row.value
    if convention == "dwave":
        return J, h
    elif convention == "minus_half":
        return dwave_conv_to_minus_half_convention(J, h)
    elif convention == "minus_half_plus_h":
        return dwave_conv_to_minus_half_convention(J, -h)
    else:
        raise ValueError("Wrong convention")


def read_instance_dict(path: os.PathLike, convention: str = "dwave"):
    df = pd.read_csv(path, sep=" ", header=None, comment="#", names=["i", "j", "value"])

    h = {}
    J = {}

    for row in df.itertuples():
        if row.i == row.j:
            h[row.i - 1] = row.value
        elif row.i > row.j:
            J[(row.j - 1, row.i - 1)] = row.value  # by zachować górnotrójkątność
        else:
            J[(row.i - 1, row.j - 1)] = row.value
    if convention == "dwave":
        return J, h



def dwave_conv_to_minus_half_convention(J: np.ndarray, h: np.ndarray):
    n = len(h)
    herminian_matrix = np.zeros((n, n))

    # de facto wyciągamy -1/2 przed macierz i zamieniamy ją na hermitowską
    for i in range(n):
        for j in range(i + 1, n):
            J_ij = J[i, j]
            herminian_matrix[i, j] = -J_ij
            herminian_matrix[j, i] = -J_ij

    x = np.random.choice([-1, 1], size=n)
    assert np.allclose(-2 * x @ J @ x.T, x @ herminian_matrix @ x.T)
    assert np.array_equal(herminian_matrix.T, herminian_matrix)  # wszystkie macierze są rzeczywiste

    new_external_fields = -1 * h
    return herminian_matrix, new_external_fields


def ising_to_qubo(J, h):
    bqm = BinaryQuadraticModel(h, J, vartype="SPIN")
    qubo, offset = bqm.to_qubo()
    N = bqm.num_variables
    Q = np.zeros((N, N))
    for (i, j), v in qubo.items():
        if i == j:
            Q[i, i] = v
        elif i > j:
            Q[j, i] = v
        else:
            Q[i, j] = v 
    return Q, offset


def calculate_energy(J: np.ndarray, h: np.ndarray, state: np.ndarray, convention: str = "minus_half"):
    if convention == "minus_half":
        return -1/2 * state @ J @ state.T - state @ h 
    elif convention == "dwave":
        return state @ J @ state.T + state @ h 


def calculate_energy_matrix(J: np.ndarray, h: np.ndarray, state: np.ndarray, convention: str = "minus_half"):
    n, _ = J.shape
    if convention == "minus_half":
        A = np.multiply(-1/2, J)
        B = np.matmul(A, state) - h.reshape(n, 1)
        C = np.multiply(state, B)
    elif convention == "dwave":
        B = np.matmul(J, state) + h.reshape(n, 1)
        C = np.multiply(state, B)
    return np.sum(C, axis=0)




