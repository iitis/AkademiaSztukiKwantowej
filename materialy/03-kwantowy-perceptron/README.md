# 3. Kwantowy perceptron

Implementacja kwantowego modelu perceptronu na komputerach kwantowych IBM. Bramkowy model obliczeń kwantowych w praktyce: Qiskit, dostęp do IBM Quantum, szum kwantowy i mitygacja błędów pomiaru, kwantowe sieci neuronowe, algorytm Deutscha, bramki pulsowe.

**Prowadzenie:** dr hab. inż. Łukasz Pawela

Wygodniejszy w przeglądaniu spis tych materiałów, z podglądem notatników w przeglądarce, znajduje się na stronie [iitis.github.io/AkademiaSztukiKwantowej](https://iitis.github.io/AkademiaSztukiKwantowej/strona/blok-03.html).

## Slajdy i skrypty (PDF)

| Materiał | Plik | Stron |
|---|---|---|
| F. Tacchino i in., „An Artificial Neuron Implemented on an Actual Quantum Processor” (arXiv:1811.02266) — artykuł źródłowy | [`materialy-zrodlowe/1811.02266-artykul-quantum-perceptron.pdf`](materialy-zrodlowe/1811.02266-artykul-quantum-perceptron.pdf) | 8 |
| Kwantowy perceptron — komplet materiałów szkolenia w jednym pliku | [`wyklady/01-kwantowy-perceptron-komplet-materialow.pdf`](wyklady/01-kwantowy-perceptron-komplet-materialow.pdf) | 136 |
| Slajdy pomocnicze do ćwiczeń | [`wyklady/02-slajdy-pomocnicze.pdf`](wyklady/02-slajdy-pomocnicze.pdf) | 24 |

## Notatniki Jupyter

Notatniki można przeczytać w przeglądarce (bez instalacji) korzystając z podglądów na stronie projektu, albo pobrać i uruchomić lokalnie.

| Notatnik | Zawartość |
|---|---|
| [`warsztaty/00-getting-started.ipynb`](warsztaty/00-getting-started.ipynb) | Getting started |
| [`warsztaty/10-noise.ipynb`](warsztaty/10-noise.ipynb) | Building Noise Models |
| [`warsztaty/20-accessing-ibmq.ipynb`](warsztaty/20-accessing-ibmq.ipynb) | Safe token management |
| [`warsztaty/30-backend_info.ipynb`](warsztaty/30-backend_info.ipynb) | Obtaining information about your `backend` |
| [`warsztaty/45-noise-revisited.ipynb`](warsztaty/45-noise-revisited.ipynb) | Noise Model Examples |
| [`warsztaty/46-device-noise.ipynb`](warsztaty/46-device-noise.ipynb) | Introduction |
| [`warsztaty/50-mthree-basic.ipynb`](warsztaty/50-mthree-basic.ipynb) | Mthree basic |
| [`warsztaty/60-perceptron.ipynb`](warsztaty/60-perceptron.ipynb) | Definition of the sign flip |
| [`warsztaty/70-perceptron-ibmq.ipynb`](warsztaty/70-perceptron-ibmq.ipynb) | Definition of th sign flip |
| [`warsztaty/80-qnn.ipynb`](warsztaty/80-qnn.ipynb) | Data preparation |
| [`warsztaty/85-deutsch.ipynb`](warsztaty/85-deutsch.ipynb) | Deutsch's problem |
| [`warsztaty/90-pulse-gates.ipynb`](warsztaty/90-pulse-gates.ipynb) | Pulse gates |

## Jak uruchomić ćwiczenia

```bash
git clone https://github.com/iitis/AkademiaSztukiKwantowej.git
cd AkademiaSztukiKwantowej/materialy/03-kwantowy-perceptron
pip install -r warsztaty/requirements.txt
jupyter notebook
```

## Materiały źródłowe

Pliki robocze (źródła LaTeX z rysunkami, wersje robocze, historia zmian) pozostają w gałęziach autorskich: [`lp/perceptron`](https://github.com/iitis/AkademiaSztukiKwantowej/tree/lp/perceptron).

## Licencja

Apache 2.0 — patrz plik [LICENSE](../../LICENSE) w katalogu głównym repozytorium.
