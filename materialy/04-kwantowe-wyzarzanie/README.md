# 4. Kwantowe wyżarzanie kombinatorycznych problemów optymalizacyjnych

Adiabatyczne obliczenia kwantowe i algorytmy inspirowane fizyką, ze szczególnym uwzględnieniem procesora kwantowego D-Wave. Model Isinga i QUBO, kodowanie problemów (Max-Cut, kolorowanie grafów, TSP), symulowane wyżarzanie, symulowana bifurkacja, obliczenia na kartach graficznych.

**Prowadzenie:** dr hab. Bartłomiej Gardas

Wygodniejszy w przeglądaniu spis tych materiałów, z podglądem notatników w przeglądarce, znajduje się na stronie [iitis.github.io/AkademiaSztukiKwantowej](https://iitis.github.io/AkademiaSztukiKwantowej/strona/blok-04.html).

## Slajdy i skrypty (PDF)

| Materiał | Plik | Stron |
|---|---|---|
| Kwantowe wyżarzanie — komplet materiałów szkolenia w jednym pliku | [`wyklady/01-kwantowe-wyzarzanie-komplet-materialow.pdf`](wyklady/01-kwantowe-wyzarzanie-komplet-materialow.pdf) | 136 |

## Notatniki Jupyter

Notatniki można przeczytać w przeglądarce (bez instalacji) korzystając z podglądów na stronie projektu, albo pobrać i uruchomić lokalnie.

| Notatnik | Zawartość |
|---|---|
| [`warsztaty/dzien-1/00_Przygotowanie_srodowiska_pracy.ipynb`](warsztaty/dzien-1/00_Przygotowanie_srodowiska_pracy.ipynb) | Pozyskanie plików |
| [`warsztaty/dzien-1/01_Klasyczny_model_Isinga.ipynb`](warsztaty/dzien-1/01_Klasyczny_model_Isinga.ipynb) | Klasyczny model Isinga |
| [`warsztaty/dzien-1/02_Model_QUBO.ipynb`](warsztaty/dzien-1/02_Model_QUBO.ipynb) | Model QUBO |
| [`warsztaty/dzien-1/03_Przykłady_kodowania_problemow_QUBO_Ising.ipynb`](warsztaty/dzien-1/03_Przykłady_kodowania_problemow_QUBO_Ising.ipynb) | Kodowanie dyskretnych problemów optymalizacyjnych za pomocą QUBO / Isinga. |
| [`warsztaty/dzien-1/04_Algorytm_wyczerpującego_przeszukiwania.ipynb`](warsztaty/dzien-1/04_Algorytm_wyczerpującego_przeszukiwania.ipynb) | Algorytm wyczerpującego przeszukiwania |
| [`warsztaty/dzien-1/05_Przegląd_algorytmow_hurystycznych.ipynb`](warsztaty/dzien-1/05_Przegląd_algorytmow_hurystycznych.ipynb) | Gwarancja optymalności |
| [`warsztaty/dzien-1/06_Kwantowy_model_Isinga.ipynb`](warsztaty/dzien-1/06_Kwantowy_model_Isinga.ipynb) | Kwantowy model Isinga |
| [`warsztaty/dzien-1/07_Twierdzenie_adiabatyczne_i_wyzarzanie_kwantowe.ipynb`](warsztaty/dzien-1/07_Twierdzenie_adiabatyczne_i_wyzarzanie_kwantowe.ipynb) | Twierdzenie adiabatyczne |
| [`warsztaty/dzien-1/08_D_Wave.ipynb`](warsztaty/dzien-1/08_D_Wave.ipynb) | Wyżarzacze kwantowe |
| [`warsztaty/dzien-1/09_zadania_dodatkowe.ipynb`](warsztaty/dzien-1/09_zadania_dodatkowe.ipynb) | Zadania dodatkowe niewymagające programowania |
| [`warsztaty/dzien-2/01_Praca_z_wyżarzaczami_kwantowymi.ipynb`](warsztaty/dzien-2/01_Praca_z_wyżarzaczami_kwantowymi.ipynb) | Ocean Software |
| [`warsztaty/dzien-2/02_Algorytm_symulowanego_wyżarzania.ipynb`](warsztaty/dzien-2/02_Algorytm_symulowanego_wyżarzania.ipynb) | Algorytm symulowanego wyżarzania |
| [`warsztaty/dzien-2/03_Algorytm_symulowanej_bifurkacji.ipynb`](warsztaty/dzien-2/03_Algorytm_symulowanej_bifurkacji.ipynb) | Algorytm symulowanej bifurkacji |
| [`warsztaty/dzien-2/04_Algorytm_wyżarzania_równoległego.ipynb`](warsztaty/dzien-2/04_Algorytm_wyżarzania_równoległego.ipynb) | Wyżarzanie równoległe |
| [`warsztaty/dzien-2/05_Algorytm_heurystycznego_branch_and_bound.ipynb`](warsztaty/dzien-2/05_Algorytm_heurystycznego_branch_and_bound.ipynb) | Branch and Bound |
| [`warsztaty/dzien-2/06_GPU.ipynb`](warsztaty/dzien-2/06_GPU.ipynb) | Wykorzystanie procesorów graficznych (GPU) w algorytmach heurystycznych |
| [`warsztaty/dzien-2/benchmarks/obliczenia_dodatkowe_srodowisko.ipynb`](warsztaty/dzien-2/benchmarks/obliczenia_dodatkowe_srodowisko.ipynb) | Generowanie danych dla benchmarków |
| [`warsztaty/dzien-2/benchmarks/obliczenia_podstawowe_srodowisko.ipynb`](warsztaty/dzien-2/benchmarks/obliczenia_podstawowe_srodowisko.ipynb) | Generowanie danych dla benchmarków |
| [`warsztaty/dzien-2/benchmarks/wyniki.ipynb`](warsztaty/dzien-2/benchmarks/wyniki.ipynb) | Wyniki benchmarków |
| [`warsztaty/pliki_pomocnicze/oscylator_harmoniczny_wyprowadzenie.ipynb`](warsztaty/pliki_pomocnicze/oscylator_harmoniczny_wyprowadzenie.ipynb) | Wyprowadzenie wzorów: |
| [`warsztaty/pliki_pomocnicze/roznica_energii_wyprowadzenie.ipynb`](warsztaty/pliki_pomocnicze/roznica_energii_wyprowadzenie.ipynb) | Wyprowadzenie wzoru na różnicę energii |
| [`warsztaty/pliki_pomocnicze/rozwiazania_wybranych_zadan_dodatkowych.ipynb`](warsztaty/pliki_pomocnicze/rozwiazania_wybranych_zadan_dodatkowych.ipynb) | Problem podziału grafu |
| [`warsztaty/pliki_pomocnicze/rozwiazywanie_rownan_rozniczkowych.ipynb`](warsztaty/pliki_pomocnicze/rozwiazywanie_rownan_rozniczkowych.ipynb) | Metoda Eulera |
| [`warsztaty/pliki_pomocnicze/wprowadzenie_do_teorii_grafow.ipynb`](warsztaty/pliki_pomocnicze/wprowadzenie_do_teorii_grafow.ipynb) | Łagodne wprowadzenie do teorii grafów |

## Jak uruchomić ćwiczenia

```bash
git clone https://github.com/iitis/AkademiaSztukiKwantowej.git
cd AkademiaSztukiKwantowej/materialy/04-kwantowe-wyzarzanie
pip install -r warsztaty/requirements.txt
jupyter notebook
```

## Materiały źródłowe

Pliki robocze (źródła LaTeX z rysunkami, wersje robocze, historia zmian) pozostają w gałęziach autorskich: [`bg/quantum_annealing`](https://github.com/iitis/AkademiaSztukiKwantowej/tree/bg/quantum_annealing).

## Licencja

Apache 2.0 — patrz plik [LICENSE](../../LICENSE) w katalogu głównym repozytorium.
