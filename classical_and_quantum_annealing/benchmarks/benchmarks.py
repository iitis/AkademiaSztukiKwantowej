import os
import numpy as np
import cupy as cp

from math import sqrt, inf
from typing import Optional
from tqdm import tqdm
from itertools import product


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def calculate_energy_matrix(J: np.ndarray, h: np.ndarray, state: np.ndarray):
    n, _ = J.shape
    A = np.multiply(-1/2, J)
    B = np.matmul(A, state) - h.reshape(n, 1)
    C = np.multiply(state, B)
    return np.sum(C, axis=0)


def calculate_energy_gpu(J: cp.ndarray, h: cp.ndarray, state: cp.ndarray):
    # Zakładamy, że J jest hermitowska z czynnikiem 1/2
    n, _ = J.shape
    A = cp.multiply(-1/2, J)
    B = cp.matmul(A, state) - h.reshape(n, 1)
    C = cp.multiply(state, B)
    return cp.sum(C, axis=0)

def wall(x: np.ndarray, y: np.ndarray):
    mask = np.abs(x) > 1
    x[mask] = np.sign(x[mask])
    y[mask] = 0
    return x, y


def balistic_simulated_bifurcation(J, h, num_steps, time_step, num_trajectories: int, 
                                   a_0: Optional[float] = None, c_0_scaling: Optional[float] = None):
    if a_0 is None:
        a_0 = 1

    N, _ = J.shape
    mean_J = np.sqrt(np.sum(np.square(J)) / (N * (N - 1)) )
    c_0 = 0.5 / (mean_J * sqrt(N))

    if c_0_scaling is not None:
        c_0 *= c_0_scaling


    a = np.linspace(0, a_0, num=num_steps)

    x = np.zeros((N, num_trajectories))
    y = np.random.uniform(-0.1, 0.1, (N, num_trajectories))

    for t in tqdm(range(num_steps), desc="Symulowana Bifurkacja"):
        y += (-1 * (a_0 - a[t]) * x + c_0 * (J @ x + h.reshape((N, 1)))) * time_step  # x(t)
        x += a_0 * y * time_step # x(t + 1)

        x, y = wall(x, y)

    x = np.sign(x)
    return x, calculate_energy_matrix(J, h, x)


def discrete_simulated_bifurcation(J, h, num_steps, time_step, num_trajectories: int, 
                                   a_0: Optional[float] = None, c_0_scaling: Optional[float] = None):
    if a_0 is None:
        a_0 = 1

    N, _ = J.shape
    mean_J = np.sqrt(np.sum(np.square(J)) / (N * (N - 1)) )
    c_0 = 0.5 / (mean_J * sqrt(N))

    if c_0_scaling is not None:
        c_0 *= c_0_scaling
        
    a = np.linspace(0, a_0, num=num_steps)

    x = np.zeros((N, num_trajectories))
    y = np.random.uniform(-0.1, 0.1, (N, num_trajectories))

    for t in tqdm(range(num_steps), desc="Symulowana Bifurkacja"):
        y += (-1 * (a_0 - a[t]) * x + c_0 * (J @ np.sign(x) + h.reshape((N, 1)))) * time_step  # y(t+1); x(t), x(t)
        x += a_0 * y * time_step # x(t + 1); y(t+1)

        x, y = wall(x, y)

    x = np.sign(x)
    return x, calculate_energy_matrix(J, h, x)


def select_lowest(energies, states, num_states):
    
    indices = cp.argpartition(energies, num_states)[:num_states]

    low_energies = energies[indices]
    low_states = states[indices]
    return low_energies, low_states

def sort_by_key(energies, states, num_states):
    order = cp.argsort(energies)[:num_states]

    low_energies = energies[order]
    low_states = states[order]
    return low_energies, low_states


def brute_force_gpu(Q,  num_states: int, sweep_size_exponent: int = 10, threadsperblock: int = 256):
    N, _ = Q.shape
    if N > 64:
        raise ValueError("Za wysoka wartość N. Ta implementacja wspiera co najwyżej 64 spiny (64-bitowy integer)")
    sweep_size = 2**sweep_size_exponent
    num_chunks = 2**(N-sweep_size_exponent)

   
    brute_force_kernel = cp.RawModule(path=os.path.join(ROOT, "cuda_kernels", "brute_force_kernel.ptx"))
    compute_energies = brute_force_kernel.get_function("compute_energies")

    blockspergrid = sweep_size//threadsperblock
    
    final_energies = cp.array([])
    final_states = cp.array([])
    
    for i in tqdm(range(num_chunks), desc="wyczerpujące przeszukiwanie"):

        energies = cp.empty(sweep_size, dtype=cp.float32)
        states = cp.empty(sweep_size, dtype=cp.int64)
        compute_energies((blockspergrid,), (threadsperblock,), 
                         (Q, cp.int32(N), cp.int32(sweep_size_exponent), energies, states, cp.int64(i)))

        low_energies, low_states = select_lowest(energies, states, num_states)
        
        final_energies = cp.concatenate((final_energies, low_energies))
        final_states = cp.concatenate((final_states, low_states))
        
        if i != 0:
            final_energies, final_states = select_lowest(final_energies, final_states, num_states)

    return sort_by_key(final_energies, final_states, num_states)


def wall_gpu(x: cp.ndarray, y: cp.ndarray):
    mask = cp.abs(x) > 1
    x[mask] = cp.sign(x[mask])
    y[mask] = 0
    return x, y


def balistic_simulated_bifurcation_gpu_naive(J, h, num_steps, time_step, num_trajectories: int, 
                                   a_0: Optional[float] = None, c_0_scaling: Optional[float] = None):
    if a_0 is None:
        a_0 = 1

    N, _ = J.shape
    mean_J = cp.sqrt(cp.sum(cp.square(J)) / (N * (N - 1)) )
    c_0 = 0.5 / (mean_J * sqrt(N))

    if c_0_scaling is not None:
        c_0 *= c_0_scaling


    a = cp.linspace(0, a_0, num=num_steps)

    x = cp.zeros((N, num_trajectories))
    y = cp.random.uniform(-0.1, 0.1, (N, num_trajectories))

    for t in tqdm(range(num_steps), desc="symulowana bifurkacja"):
        y += (-1 * (a_0 - a[t]) * x + c_0 * (cp.matmul(J,x)) + h.reshape((N, 1))) * time_step  # x(t)
        x += a_0 * y * time_step # x(t + 1)

        x, y = wall_gpu(x, y)

    x = cp.sign(x)
    return x, calculate_energy_gpu(J, h, x)


def balistic_simulated_bifurcation_gpu(J, h, num_steps, time_step, num_trajectories: int, 
                                   a_0: Optional[float] = None, c_0_scaling: Optional[float] = None):
    if a_0 is None:
        a_0 = 1
    
    N, _ = J.shape
    mean_J = np.sqrt(np.sum(np.square(J)) / (N * (N - 1)) )
    c_0 = 0.5 / (mean_J * sqrt(N))

    if c_0_scaling is not None:
        c_0 *= c_0_scaling

    dtype = cp.float32

    a = [dtype(i * a_0 / (num_steps - 1)) for i in range(num_steps)]
    
    a_0 = dtype(a_0)
    c_0 = dtype(c_0.item())
    time_step = dtype(time_step)

    x = cp.zeros((N, num_trajectories), dtype=dtype)
    y = cp.random.uniform(-0.1, 0.1, (N, num_trajectories), dtype=dtype)

    x_new = cp.empty_like(x)
    y_new = cp.empty_like(y)

    threadsperblock = 256  # Ilość wątków w bloku,
    blockspergrid_x = num_trajectories  # każdy blok zajmuje się trajektorią
    blockspergrid_y = (N + threadsperblock - 1) // threadsperblock  # wystarczająca ilość bloków by pomieścić całą kolumnę 
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    kernel = cp.RawModule(path=os.path.join(ROOT, "cuda_kernels", "sbm_kernel.ptx"))
    update_y = kernel.get_function("update_y")
    update_x_and_wall = kernel.get_function("update_x_and_wall")

    for t in tqdm(range(num_steps), desc="balistyczna symulowana bifurkacja"):
        A = cp.matmul(J, x) + h.reshape((N, 1))
        update_y(blockspergrid, (threadsperblock,),(x, y, A, 
                                                    a_0, a[t], c_0, time_step, N,
                                                    y_new))
        y = y_new
        update_x_and_wall(blockspergrid, (threadsperblock,),(x, y, 
                                                             a_0, time_step, N, 
                                                             x_new, y_new))
        x = x_new
        y = y_new

    x = cp.sign(x)
    return x, calculate_energy_gpu(J, h, x)


def discrete_simulated_bifurcation_gpu_naive(J, h, num_steps, time_step, num_trajectories: int, 
                                   a_0: Optional[float] = None, c_0_scaling: Optional[float] = None):
    if a_0 is None:
        a_0 = 1

    N, _ = J.shape
    mean_J = cp.sqrt(cp.sum(cp.square(J)) / (N * (N - 1)) )
    c_0 = 0.5 / (mean_J * sqrt(N))

    if c_0_scaling is not None:
        c_0 *= c_0_scaling


    a = cp.linspace(0, a_0, num=num_steps)

    x = cp.zeros((N, num_trajectories))
    y = cp.random.uniform(-0.1, 0.1, (N, num_trajectories))

    for t in tqdm(range(num_steps), desc="symulowana bifurkacja"):
        y += (-1 * (a_0 - a[t]) * x + c_0 * (cp.matmul(J,cp.sign(x)) + h.reshape((N, 1)))) * time_step  # x(t)
        x += a_0 * y * time_step # x(t + 1)

        x, y = wall_gpu(x, y)

    x = cp.sign(x)
    return x, calculate_energy_gpu(J, h, x)


def discrete_simulated_bifurcation_gpu(J, h, num_steps, time_step, num_trajectories: int, 
                                   a_0: Optional[float] = None, c_0_scaling: Optional[float] = None):
    if a_0 is None:
        a_0 = 1
    
    N, _ = J.shape
    mean_J = np.sqrt(np.sum(np.square(J)) / (N * (N - 1)) )
    c_0 = 0.5 / (mean_J * sqrt(N))

    if c_0_scaling is not None:
        c_0 *= c_0_scaling

    dtype = cp.float32

    a = [dtype(i * a_0 / (num_steps - 1)) for i in range(num_steps)]
    
    a_0 = dtype(a_0)
    c_0 = dtype(c_0.item())
    time_step = dtype(time_step)

    x = cp.zeros((N, num_trajectories), dtype=dtype)
    y = cp.random.uniform(-0.1, 0.1, (N, num_trajectories), dtype=dtype)

    x_new = cp.empty_like(x)
    y_new = cp.empty_like(y)

    threadsperblock = 256  # Ilość wątków w bloku,
    blockspergrid_x = num_trajectories  # każdy blok zajmuje się trajektorią
    blockspergrid_y = (N + threadsperblock - 1) // threadsperblock  # wystarczająca ilość bloków by pomieścić całą kolumnę 
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    kernel = cp.RawModule(path=os.path.join(ROOT, "cuda_kernels", "sbm_kernel.ptx"))
    update_y = kernel.get_function("update_y")
    update_x_and_wall = kernel.get_function("update_x_and_wall")

    for t in tqdm(range(num_steps), desc="dyskretna symulowana bifurkacja"):
        sign_x = cp.sign(x)
        A = cp.matmul(J, sign_x) + h.reshape((N, 1))
        update_y(blockspergrid, (threadsperblock,),(x, y, A, 
                                                    a_0, a[t], c_0, time_step, N,
                                                    y_new))
        y = y_new
        update_x_and_wall(blockspergrid, (threadsperblock,),(x, y, 
                                                             a_0, time_step, N,
                                                             x_new, y_new))
        x = x_new
        y = y_new

    x = cp.sign(x)
    return x, calculate_energy_gpu(J, h, x)


def calculate_energy(J: np.ndarray, h: np.ndarray, state: np.ndarray, convention: str = "minus_half"):
    if convention == "minus_half":
        return -1/2 * state @ J @ state.T - state @ h 
    elif convention == "dwave":
        return state @ J @ state.T + state @ h 
    

def brute_force_naive(J, h):
    best_energy = inf
    best_state = None
    n = len(h)

    for state in tqdm(product([-1, 1], repeat=n), desc="Wyczerpujące przeszukiwanie", total=2**n):
        state = np.array(state)
        energy = calculate_energy(J, h, state, convention="dwave")
        if energy < best_energy:
            best_energy = energy
            best_state = state

    return best_state, best_energy