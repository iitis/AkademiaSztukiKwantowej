# Pegasus CBFM-P

import os
import numpy as np
import dwave_networkx as dnx



rng = np.random.default_rng()
P = 2
save_path = os.path.join("instancje", "Pegasus", f"P{P}_CBFM-P.txt")

graph = dnx.pegasus_graph(P, nice_coordinates=True)

def nice_to_spin_glass(node: tuple, size: int) -> int:
    t, y, x, u, k = node
    if u == 1:
        a = 4 + k + 1
    else:
        a = abs(k - 3) + 1
    b = abs(y - (size - 2))

    spin_glas_linear = 8 * t + 24 * x + 24 * (size - 1) * b + a
    return spin_glas_linear



bias = {nice_to_spin_glass(node, P) : rng.choice([-1, 0], p=[0.85, 0.15]) for node in graph.nodes()}
bias = dict(sorted(bias.items()))
couplings = {
    (nice_to_spin_glass(e1, P), nice_to_spin_glass(e2, P)): rng.choice([-1, 0, 1], p=[0.1, 0.35, 0.55])
    for (e1, e2) in graph.edges()
}
couplings = dict(sorted(couplings.items()))



with open(save_path, "w") as f:
    for node, value in bias.items():
        f.write(f"{node} {node} {value.item()}\n")
    for (e1, e2), value in couplings.items():
        f.write(f"{e1} {e2} {value.item()}\n")