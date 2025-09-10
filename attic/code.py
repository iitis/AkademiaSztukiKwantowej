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