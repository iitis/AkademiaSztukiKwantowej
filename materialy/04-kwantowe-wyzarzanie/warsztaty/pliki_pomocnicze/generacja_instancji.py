import os
import numpy as np
import dwave_networkx as dnx
import networkx as nx

from typing import Optional, Callable
from dwave.system import DWaveSampler


cwd = os.getcwd()
rng = np.random.default_rng()


def nice_to_spin_glass(node: tuple, size: int) -> int:
    t, y, x, u, k = node
    if u == 1:
        a = 4 + k + 1
    else:
        a = abs(k - 3) + 1
    b = abs(y - (size - 2))

    spin_glas_linear = 8 * t + 24 * x + 24 * (size - 1) * b + a
    return spin_glas_linear


def generate_cbfm_p_instance(P: int, graph):

    bias = {nice_to_spin_glass(node, P) : rng.choice([-1, 0], p=[0.85, 0.15]) for node in graph.nodes()}
    bias = dict(sorted(bias.items()))
    couplings = {
        (nice_to_spin_glass(e1, P), nice_to_spin_glass(e2, P)): rng.choice([-1, 0, 1], p=[0.1, 0.35, 0.55])
        for (e1, e2) in graph.edges()
    }
    couplings = dict(sorted(couplings.items()))
    return couplings, bias


def generate_random_instance(graph, transform: Optional[Callable] = None, *args):
    if transform is None:
        transform = lambda x: x 

    bias = {transform(node, *args): rng.choice(np.linspace(-1, 1, num=9, endpoint=True)) 
            for node in graph.nodes()}
    bias = dict(sorted(bias.items()))
    couplings = {
        (transform(e1, *args), transform(e2, *args)): rng.choice(np.linspace(-1, 1, num=9, endpoint=True))
        for (e1, e2) in graph.edges()
    }
    couplings = dict(sorted(couplings.items()))
    return couplings, bias


def save_instance(couplings: dict, bias: dict, save_path: os.PathLike):
    with open(save_path, "w") as f:
        for node, value in bias.items():
            f.write(f"{node} {node} {value.item()}\n")
        for (e1, e2), value in couplings.items():
            f.write(f"{e1} {e2} {value.item()}\n")


def grid_to_linear(node: tuple[int, int], grid_size: int):
    x, y = node
    return x*grid_size + y + 1

if __name__ == "__main__":
    sampler = DWaveSampler(solver="Advantage_system4.1")
    graph = sampler.to_networkx_graph()
    
    J, h = generate_random_instance(graph, lambda x: x+1)
    save_instance(J, h, os.path.join("instancje", "Pegasus", "Advantage_system4.1_random.txt"))


   
    