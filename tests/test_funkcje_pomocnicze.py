"""
Testy jednostkowe dla funkcje_pomocnicze.py.

Dlaczego testy?
- Funkcje konwertujące Ising ↔ QUBO muszą być matematycznie poprawne;
  błąd tu propaguje się przez WSZYSTKIE notebooki.
- Testy dokumentują oczekiwane zachowanie i łapią regresje po zmianach.
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Kwantowe_wyzarzanie_kombinatorycznych_problemow_optymalizacyjnych.pliki_pomocnicze.funkcje_pomocnicze import (
    ising_to_qubo,
    calculate_energy,
    calculate_energy_matrix,
    dwave_conv_to_minus_half_convention,
)


# ---------------------------------------------------------------------------
# Pomocnicze
# ---------------------------------------------------------------------------

def qubo_energy(Q: np.ndarray, x: np.ndarray) -> float:
    """E_QUBO = x^T Q x"""
    return float(x @ Q @ x)


def ising_energy_dwave(J: np.ndarray, h: np.ndarray, s: np.ndarray) -> float:
    """E_Ising = sum_{i<j} J_ij s_i s_j + sum_i h_i s_i  (konwencja D-Wave)"""
    return float(s @ J @ s + s @ h)


# ---------------------------------------------------------------------------
# ising_to_qubo
# ---------------------------------------------------------------------------

class TestIsingToQubo:
    """Sprawdza, czy konwersja Ising → QUBO zachowuje energię.

    Idea: dla każdego stanu Isinga s ∈ {-1,+1}^n
    i odpowiadającego stanu binarnego x = (s+1)/2 ∈ {0,1}^n
    musi zachodzić: E_Ising(s) == E_QUBO(x) + offset.
    """

    def _check_all_states(self, J, h):
        Q, offset = ising_to_qubo(J, h)
        n = len(h)
        for bits in range(2**n):
            x = np.array([(bits >> i) & 1 for i in range(n)], dtype=float)
            s = 2 * x - 1
            e_ising = ising_energy_dwave(J, h, s)
            e_qubo = qubo_energy(Q, x) + offset
            assert np.isclose(e_ising, e_qubo), (
                f"s={s}, x={x}: E_Ising={e_ising:.4f} != E_QUBO+offset={e_qubo:.4f}"
            )

    def test_trivial_single_spin(self):
        """n=1: J=0, h=1  →  E(-1)=-1, E(+1)=+1"""
        J = np.zeros((1, 1))
        h = np.array([1.0])
        self._check_all_states(J, h)

    def test_two_spins_ferromagnet(self):
        """n=2: J[0,1]=1 (sprzężenie ferromagnetyczne)"""
        J = np.array([[0.0, 1.0],
                      [0.0, 0.0]])
        h = np.zeros(2)
        self._check_all_states(J, h)

    def test_two_spins_antiferromagnet(self):
        """n=2: J[0,1]=-1 (sprzężenie antyferromagnetyczne)"""
        J = np.array([[0.0, -1.0],
                      [0.0,  0.0]])
        h = np.zeros(2)
        self._check_all_states(J, h)

    def test_two_spins_with_fields(self):
        """n=2: sprzężenie + pola zewnętrzne"""
        J = np.array([[0.0, 0.5],
                      [0.0, 0.0]])
        h = np.array([-0.3, 0.7])
        self._check_all_states(J, h)

    def test_three_spins_random(self):
        """n=3: losowa instancja z ustalonym seedem"""
        rng = np.random.default_rng(42)
        n = 3
        J = np.triu(rng.uniform(-1, 1, (n, n)), k=1)
        h = rng.uniform(-1, 1, n)
        self._check_all_states(J, h)

    def test_four_spins_random(self):
        """n=4: losowa instancja"""
        rng = np.random.default_rng(7)
        n = 4
        J = np.triu(rng.uniform(-1, 1, (n, n)), k=1)
        h = rng.uniform(-1, 1, n)
        self._check_all_states(J, h)

    def test_returns_upper_triangular(self):
        """Q musi być górnotrójkątna (taka sama konwencja co J)"""
        J = np.array([[0.0, 1.0], [0.0, 0.0]])
        h = np.zeros(2)
        Q, _ = ising_to_qubo(J, h)
        assert Q[1, 0] == 0.0, "Q nie jest górnotrójkątna"

    def test_offset_type(self):
        J = np.array([[0.0, 1.0], [0.0, 0.0]])
        h = np.array([0.5, -0.5])
        _, offset = ising_to_qubo(J, h)
        assert isinstance(float(offset), float)


# ---------------------------------------------------------------------------
# calculate_energy
# ---------------------------------------------------------------------------

class TestCalculateEnergy:
    """Sprawdza obliczanie energii w obu konwencjach."""

    def test_minus_half_convention_known_case(self):
        """
        Konwencja minus_half: E = -1/2 s^T J_herm s - s @ h
        Dla J_herm = [[0,2],[2,0]], h=[0,0], s=[1,1]:
          E = -1/2 * (1*0*1 + 1*2*1 + 1*2*1 + 1*0*1) - 0 = -1/2 * 4 = -2
        """
        J = np.array([[0.0, 2.0],
                      [2.0, 0.0]])
        h = np.zeros(2)
        s = np.array([1.0, 1.0])
        e = calculate_energy(J, h, s, convention="minus_half")
        assert np.isclose(e, -2.0)

    def test_dwave_convention_known_case(self):
        """
        Konwencja dwave: E = s^T J s + s @ h
        Dla J = [[0,1],[0,0]], h=[0,0], s=[1,1]:
          E = 1*0*1 + 1*1*1 + 1*0*1 + 1*0*1 + 0 = 1
        """
        J = np.array([[0.0, 1.0],
                      [0.0, 0.0]])
        h = np.zeros(2)
        s = np.array([1.0, 1.0])
        e = calculate_energy(J, h, s, convention="dwave")
        assert np.isclose(e, 1.0)

    def test_calculate_energy_vs_matrix(self):
        """calculate_energy i calculate_energy_matrix muszą dawać ten sam wynik."""
        rng = np.random.default_rng(0)
        n = 5
        J = np.triu(rng.uniform(-1, 1, (n, n)), k=1)
        J = J + J.T  # symetryczna dla konwencji minus_half
        h = rng.uniform(-1, 1, n)

        # pojedynczy stan
        s = rng.choice([-1.0, 1.0], size=n)
        e_single = calculate_energy(J, h, s, convention="minus_half")

        # macierzowo (wiele stanów jednocześnie — tutaj jeden stan jako kolumna)
        states_matrix = s.reshape(n, 1)
        e_matrix = calculate_energy_matrix(J, h, states_matrix, convention="minus_half")
        assert np.isclose(e_single, e_matrix[0])


# ---------------------------------------------------------------------------
# dwave_conv_to_minus_half_convention
# ---------------------------------------------------------------------------

class TestDwaveConvToMinusHalf:
    """Sprawdza, czy konwersja konwencji zachowuje energię."""

    def test_energy_preserved(self):
        """
        Po konwersji: E_minus_half(J_herm, h_new, s) == E_dwave(J, h, s)
        E_dwave   = s^T J s + s @ h
        E_minus_half = -1/2 s^T J_herm s - s @ h_new
        """
        rng = np.random.default_rng(3)
        n = 4
        J = np.triu(rng.uniform(-1, 1, (n, n)), k=1)
        h = rng.uniform(-1, 1, n)

        J_herm, h_new = dwave_conv_to_minus_half_convention(J, h)

        for _ in range(20):
            s = rng.choice([-1.0, 1.0], size=n)
            e_dwave = calculate_energy(J, h, s, convention="dwave")
            e_new = calculate_energy(J_herm, h_new, s, convention="minus_half")
            assert np.isclose(e_dwave, e_new), (
                f"Energia nie zachowana dla s={s}: {e_dwave} != {e_new}"
            )

    def test_result_is_symmetric(self):
        """J_herm musi być macierzą symetryczną."""
        rng = np.random.default_rng(5)
        n = 5
        J = np.triu(rng.uniform(-1, 1, (n, n)), k=1)
        h = rng.uniform(-1, 1, n)
        J_herm, _ = dwave_conv_to_minus_half_convention(J, h)
        assert np.allclose(J_herm, J_herm.T)
