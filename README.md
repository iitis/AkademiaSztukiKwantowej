# Akademia Sztuki Kwantowej — Kwantowe Wyżarzanie

Materiały do 2-dniowego szkolenia z kwantowego wyżarzania (quantum annealing)
i kombinatorycznych problemów optymalizacyjnych.

## Zawartość kursu

```
Kwantowe_wyzarzanie_kombinatorycznych_problemow_optymalizacyjnych/
├── Dzien_1/   — teoria (model Isinga, QUBO, twierdzenie adiabatyczne, D-Wave)
├── Dzien_2/   — praktyka (SA, symulowana bifurkacja, branch & bound, GPU)
└── pliki_pomocnicze/   — moduły Python + instancje testowe
```

**Dzień 1** (teoria):
1. Przygotowanie środowiska
2. Klasyczny model Isinga
3. Model QUBO — kodowanie problemów
4. Przykłady: Max-Cut, kolorowanie grafu, TSP
5. Algorytm wyczerpującego przeszukiwania (brute force)
6. Przegląd algorytmów heurystycznych
7. Kwantowy model Isinga
8. Twierdzenie adiabatyczne i wyżarzanie kwantowe
9. Procesor D-Wave

**Dzień 2** (praktyka):
1. Praca z wyżarzaczami kwantowymi (D-Wave Ocean SDK)
2. Symulowane wyżarzanie (SA)
3. Symulowana bifurkacja (SB)
4. Wyżarzanie równoległe
5. Heurystyczny branch & bound
6. Implementacja GPU (CuPy)

## Szybki start

```bash
# 1. Sklonuj repozytorium
git clone https://github.com/iitis/AkademiaSztukiKwantowej.git
cd AkademiaSztukiKwantowej

# 2. Zainstaluj zależności do większości notebooków
pip install -r requirements.txt

# 3. Uruchom Jupyter
jupyter notebook
```

> Notebooki korzystające z D-Wave wymagają konta na https://cloud.dwavesys.com
> i skonfigurowanego tokenu (`dwave config create`).
>
> Notebook GPU (`Dzien_2/06_GPU.ipynb`) wymaga osobnej instalacji CuPy
> dopasowanej do lokalnej wersji CUDA.
>
> Notebooki z katalogu `Dzien_2/benchmarks/` wymagają dodatkowego środowiska
> eksperymentalnego (GPU, CuPy/PyTorch CUDA oraz zewnętrzne biblioteki takie
> jak `simulated_bifurcation`, `omnisolver`, `scikit-learn`) i nie są częścią
> domyślnego smoke testu.

## Testy

```bash
pip install pytest
pytest tests/ -v
```

Testy weryfikują poprawność matematyczną funkcji konwertujących
Ising ↔ QUBO oraz obliczania energii.

## Struktura plików pomocniczych

| Plik | Co robi |
|------|---------|
| `funkcje_pomocnicze.py` | Wczytywanie instancji, konwersja Ising↔QUBO, obliczanie energii |
| `funkcje_pomocnicze_gpu.py` | Wersja GPU (CuPy) obliczania energii |
| `generacja_instancji.py` | Generowanie losowych instancji Isinga na grafach Pegasus/Grid |
| `instancje/` | Gotowe instancje testowe (P2–P16, Grid5–Grid100, K8) |

## CI

GitHub Actions uruchamia automatycznie:
- **testy jednostkowe** przy każdym push
- **nieblokujący smoke test notebooków** przy push do `master` / `main`
  oraz przy PR do głównej gałęzi

Smoke test pomija notebooki wymagające zewnętrznego środowiska
(D-Wave Cloud, GPU/CuPy oraz katalog `Dzien_2/benchmarks`), więc jego celem
jest szybkie wykrywanie regresji uruchomieniowych w pozostałych materiałach.

## Licencja

Apache 2.0 — szczegóły w pliku [LICENSE](LICENSE).
