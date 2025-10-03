## Przykładowy problem podziału
## rozwiążemy go metodą wyczerpującego przeszukiwania - sprawdzanie wszystkich możliwości
import numpy as np
from itertools import product
from math import inf

# zadany zbiór
S = [2, 3, 4, 5, 6] 

def calculate_hamiltonian(S, spins, A = 1):
    return A * sum([S[i] * spins[i] for i in range(len(S))])**2

best_energy = inf
best_solution = None
for solution in product([-1, 1], repeat=len(S)):
    energy = calculate_hamiltonian(S, solution)
    if energy < best_energy:
        best_energy = energy
        best_solution = solution

S_1 = []
S_2 = []
for idx, spin in enumerate(best_solution):
    if spin == 1:
        S_1.append(S[idx])
    elif spin == -1:
        S_2.append(S[idx])
    else:
        raise ValueError("Coś zlego się stało ze spinem")

print("Najlepsza znaleziona wartość Hamiltonianu: ", best_energy)
print("Rozwiązanie: ", best_solution)
print("Podział: ", S_1, S_2)


solution = inf
for x in product([0, 1], repeat=4):
    x = np.array(x)
    new_solution = x @ Q @ x.T
    if new_solution < solution:
        solution = deepcopy(new_solution)
        solution_vector = deepcopy(x)

print("rozwiązanie:", end = " ")
for idx, i in enumerate(solution_vector):
    print(f"x_{idx + 1} = {i},", end = " ")
print(f"\ny_min = {solution}")


def random_neightbour(conf: np.ndarray):
    new_conf = deepcopy(conf)
    flip_idx = np.random.randint(len(new_conf))
    new_conf[flip_idx] *= -1  # zmieniamy spin losowego wierzchołka
    return new_conf


import networkx as nx
import numpy as np
from itertools import product
from math import inf
# tworzymy losowy graf o parzystej liczbie wierzchołków

graph = nx.erdos_renyi_graph(n=6, p=0.3, seed=2137)

h = [0 for _ in graph.nodes]
m = nx.to_numpy_array(graph)
nx.draw(graph, with_labels=True)
for i in range(graph.number_of_nodes()):
    for j in range(graph.number_of_nodes()):
        if i != j:
            m[i, j] = 1 if m[i,j] == 0 else 0.5
            
best_energy = inf
for state in product([-1, 1], repeat=graph.number_of_nodes()):
    s = np.array(state)
    energy = s @ m @ s.T
    if energy <= best_energy:
        best_energy = energy
        best_state = state
print(best_energy)
print(best_state)


import numpy as np
from math import inf
from itertools import product

rng = np.random.default_rng(seed=42)

def random_subset(s):
    return {x for x in s if rng.choice((True, False))}

def check_solution(x, U, R):
    sol = 0
    for alpha in U:
        # tworzymy i:alpha in V_i
        K = []
        for i in range(N):
            if alpha in R[i]:
                K.append(i)

        temp = 0
        for i in K:
            temp += x[i]
        sol += (1 - temp)**2

    if sol ==0:
        return "Pełne pokrycie"
    else:
        return "Błąd"

n = 4
N = 5

U = set(range(1, n+1))
R = [random_subset(U) for i in range(N)]

Q = np.zeros((N,N))

for i in range(N):
    for j in range(N):
        if i==j:
            Q[i,j] = -n
        else:
            Q[i,j] = 2*len(R[i] & R[j])


print(Q)
best_energy = inf
for state in product([0, 1], repeat=N):
    x = np.array(state)
    energy = x @ Q @ x
    if energy <= best_energy:
        best_energy = energy
        best_state = state


print("x = ", best_state)
print("R =", R)

check_solution([0, 1, 0, 1, 1], U, R)


# E = -12772
# best found -12742 steps 10^4 trajectories 2^10 time approx 43s
# najperw trzeba raz przepuscić by kernel wszedł do pamięci podręcznej (można użyć mało kroków)
from funkcje_pomocnicze import read_instance, full_pegasus
J, h = read_instance(full_pegasus.path, convention="minus_half")


J = cp.asarray(J, dtype=cp.float32)
h = cp.asarray(h, dtype=cp.float32)


states, energy = parrarel_annealing_gpu(J, h, step_size=0.01, lambda_t_max=10, num_steps=10000, num_trajectories=2**10)
print(min(energy))