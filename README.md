# Akademia Sztuki Kwantowej

Repozytorium materiałów szkoleniowych projektu [Akademia Sztuki Kwantowej](https://akademia.iitis.pl/) realizowanego przez Instytut Informatyki Teoretycznej i Stosowanej PAN w Gliwicach.

Materiały poszczególnych bloków tematycznych znajdują się w **osobnych gałęziach** — poniżej przewodnik.

## Bloki tematyczne

### 1. Obliczenia i algorytmy kwantowe

Wprowadzenie do obliczeń kwantowych — od podstaw fizyki kwantowej po zaawansowane algorytmy. Program obejmuje: stany i pomiary kwantowe, bramki kwantowe, układy złożone, informację kwantową, kwantowe gęste kodowanie, teleportację kwantową oraz kryptografię kwantową.

| Gałąź | Zawartość | Prowadzący |
|-------|-----------|------------|
| [`pg/qcintro`](https://github.com/iitis/AkademiaSztukiKwantowej/tree/pg/qcintro) | 5 wykładów (LaTeX/Beamer): motywacja do QC, model bramkowy, kwantowe sieci neuronowe, obliczenia adiabatyczne, VQE. Materiały uzupełniające (książeczka + slajdy) oraz przykłady w Pythonie (PennyLane). | dr hab. inż. Piotr Gawron |
| [`zp/QI_intro_lectures`](https://github.com/iitis/AkademiaSztukiKwantowej/tree/zp/QI_intro_lectures) | Plan 12 wykładów: przestrzenie Hilberta, superpozycja, splątanie, stany Bella, twierdzenie o nieklonowaniu, teleportacja, kryptografia kwantowa, algorytmy Shora i Grovera, korekcja błędów. **Uwaga: na razie tylko plan (`plan.md`), brak treści wykładów.** | prof. dr hab. Zbigniew Puchała |

### 2. Uczenie maszynowe

Wprowadzenie do klasycznego (nie-kwantowego) uczenia maszynowego. Nacisk na zrozumienie istoty działania metod i ich praktyczne zastosowanie — od analizy danych, przez sieci neuronowe, po klasyczne modele (SVM). Stanowi podstawę do metod kwantowych.

| Gałąź | Zawartość | Prowadzący |
|-------|-----------|------------|
| [`mlintro`](https://github.com/iitis/AkademiaSztukiKwantowej/tree/mlintro) | Wykład wprowadzający (LaTeX): sztuczne sieci neuronowe, metody jądrowe, klasyfikacja danych hiperspektralnych. | dr hab. inż. Przemysław Głomb |
| [`pg_final`](https://github.com/iitis/AkademiaSztukiKwantowej/tree/pg_final) | 5 wykładów (perceptron, MLP, metody statystyczne ML) + rozbudowane warsztaty (Jupyter): klasteryzacja, PCA/t-SNE, detekcja anomalii, SVM/SHAP, sieci konwolucyjne, LLM. | dr hab. inż. Przemysław Głomb |

### 3. Uczenie architektur kwantowych

Implementacja kwantowego modelu perceptronu przy użyciu technologii komputerów kwantowych IBM. Bramkowy model obliczeń kwantowych w praktyce — programowanie na komputerze kwantowym.

| Gałąź | Zawartość | Prowadzący |
|-------|-----------|------------|
| [`lp/perceptron`](https://github.com/iitis/AkademiaSztukiKwantowej/tree/lp/perceptron) | Szkolenie 2-dniowe (17 notebooków Jupyter): Qiskit, dostęp do IBM Quantum, szum kwantowy, korekcja błędów (M3), perceptron kwantowy, QNN, algorytm Deutscha, bramki pulsowe. | dr hab. inż. Łukasz Pawela |

### 4. Kwantowe wyżarzanie kombinatorycznych problemów optymalizacyjnych

Adiabatyczne obliczenia kwantowe i algorytmy inspirowane fizyką, ze szczególnym uwzględnieniem procesora kwantowego D-Wave. Modele QUBO/Ising, topologia procesora, ewolucja adiabatyczna.

| Gałąź | Zawartość | Prowadzący |
|-------|-----------|------------|
| [`bg/quantum_annealing`](https://github.com/iitis/AkademiaSztukiKwantowej/tree/bg/quantum_annealing) | Szkolenie 2-dniowe: dzień 1 (teoria) — model Isinga, QUBO, kodowanie problemów (Max-Cut, kolorowanie grafów, TSP), algorytmy heurystyczne, D-Wave; dzień 2 (praktyka) — symulowane wyżarzanie, symulowana bifurkacja, akceleracja GPU (CUDA). Notebooki Jupyter + moduły Python + instancje testowe (Chimera, Pegasus, Zephyr). | dr hab. Bartłomiej Gardas |

### 5. Kwantowe sieci neuronowe i kwantowe metody jądrowe

Rozwiązywanie problemów uczenia maszynowego za pomocą komputerów kwantowych. Tworzenie klasyfikatorów i regresorów działających na prawdziwych danych.

| Gałąź | Zawartość | Prowadzący |
|-------|-----------|------------|
| [`pg_final`](https://github.com/iitis/AkademiaSztukiKwantowej/tree/pg_final) | 5 wykładów (perceptron, MLP, metody statystyczne ML) + rozbudowane warsztaty (Jupyter): klasteryzacja, PCA/t-SNE, detekcja anomalii, SVM/SHAP, sieci konwolucyjne, LLM. | dr hab. inż. Przemysław Głomb |

## Harmonogram wydarzeń

Pełny kalendarz szkoleń i warsztatów: [akademia.iitis.pl](https://akademia.iitis.pl/)

## Strona projektu

Strona projektu: [ask.iitis.pl](https://ask.iitis.pl/)
