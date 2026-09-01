# -*- coding: utf-8 -*-
"""Opisy bloków tematycznych i pojedynczych materiałów.

Plik jest jedynym miejscem, w którym trzyma się teksty prezentowane na
stronie i w plikach README. Skrypt `zbuduj_strone.py` czyta go i generuje
resztę automatycznie.
"""

BLOKI = [
    dict(
        katalog="01-obliczenia-i-algorytmy-kwantowe",
        tytul="Obliczenia i algorytmy kwantowe",
        prowadzacy=["dr hab. inż. Piotr Gawron", "prof. dr hab. Zbigniew Puchała",
                    "dr Hanna Wojewódka-Ściążko"],
        opis="Wprowadzenie do obliczeń kwantowych — od podstaw matematycznych i fizycznych "
             "po algorytmy. Stany i pomiary kwantowe, bramki, układy złożone, informacja "
             "kwantowa, gęste kodowanie, teleportacja, kryptografia kwantowa oraz algorytmy "
             "Deutscha, Grovera i Shora.",
        galaz=["pg/qcintro", "zp/QI_intro_lectures"],
    ),
    dict(
        katalog="02-uczenie-maszynowe",
        tytul="Dwa dni z uczeniem maszynowym",
        prowadzacy=["dr hab. inż. Przemysław Głomb"],
        opis="Wprowadzenie do klasycznego (nie-kwantowego) uczenia maszynowego: analiza "
             "danych, klasteryzacja, redukcja wymiarowości, detekcja anomalii, maszyny "
             "wektorów nośnych, sieci neuronowe i konwolucyjne, duże modele językowe. "
             "Blok stanowi podstawę do metod kwantowych.",
        galaz=["pg_final", "mlintro"],
    ),
    dict(
        katalog="03-kwantowy-perceptron",
        tytul="Kwantowy perceptron",
        prowadzacy=["dr hab. inż. Łukasz Pawela"],
        opis="Implementacja kwantowego modelu perceptronu na komputerach kwantowych IBM. "
             "Bramkowy model obliczeń kwantowych w praktyce: Qiskit, dostęp do IBM Quantum, "
             "szum kwantowy i mitygacja błędów pomiaru, kwantowe sieci neuronowe, algorytm "
             "Deutscha, bramki pulsowe.",
        galaz=["lp/perceptron"],
    ),
    dict(
        katalog="04-kwantowe-wyzarzanie",
        tytul="Kwantowe wyżarzanie kombinatorycznych problemów optymalizacyjnych",
        prowadzacy=["dr hab. Bartłomiej Gardas"],
        opis="Adiabatyczne obliczenia kwantowe i algorytmy inspirowane fizyką, ze szczególnym "
             "uwzględnieniem procesora kwantowego D-Wave. Model Isinga i QUBO, kodowanie "
             "problemów (Max-Cut, kolorowanie grafów, TSP), symulowane wyżarzanie, symulowana "
             "bifurkacja, obliczenia na kartach graficznych.",
        galaz=["bg/quantum_annealing"],
    ),
    dict(
        katalog="05-kwantowe-uczenie-maszynowe",
        tytul="Kwantowe uczenie maszynowe i kwantowe metody jądrowe",
        prowadzacy=["dr hab. inż. Piotr Gawron"],
        opis="Rozwiązywanie problemów uczenia maszynowego za pomocą komputerów kwantowych: "
             "obwody parametryzowalne, algorytmy wariacyjne, kwantowe sieci neuronowe, "
             "kwantowe metody jądrowe, modele hybrydowe (PyTorch/JAX + PennyLane) oraz "
             "wpływ szumu na jakość modeli.",
        galaz=["pg/szkolenie"],
    ),
]

# Czytelne tytuły plików PDF (klucz: ścieżka względem katalogu bloku).
TYTULY = {
    "wyklady/01-wprowadzenie-do-obliczen-kwantowych.pdf":
        "Wprowadzenie do obliczeń kwantowych — wykład (Z. Puchała)",
    "wyklady/02-algorytmy-kwantowe.pdf":
        "Algorytmy kwantowe — wykład (Z. Puchała)",
    "wyklady/03-algorytmy-numeryczne-minimum-funkcji-celu.pdf":
        "Algorytmy numeryczne znajdujące minimum funkcji celu (P. Gawron)",
    "wyklady/04-podstawy-matematyczne-obliczen-kwantowych-i-si.pdf":
        "Podstawy matematyczne obliczeń kwantowych i sztucznej inteligencji "
        "(H. Wojewódka-Ściążko)",
    "wyklady/05-wprowadzenie-do-obliczen-kwantowych-slajdy-5-wykladow.pdf":
        "Komplet slajdów pięciu wykładów: motywacja, model bramkowy, kwantowe sieci "
        "neuronowe, model Isinga, zastosowania (P. Gawron)",
    "wyklady/06-materialy-uzupelniajace-ksiazeczka.pdf":
        "Materiały uzupełniające w formie książeczki: podstawy matematyczne, mechanika "
        "kwantowa, informacja kwantowa (P. Gawron)",
    "wyklady/01-perceptron.pdf": "Perceptron",
    "wyklady/02-sieci-mlp.pdf": "Sieci wielowarstwowe (MLP)",
    "wyklady/03-sieci-mlp-analiza.pdf": "Sieci MLP — analiza działania",
    "wyklady/04-kowariancja-i-metody-statystyczne.pdf": "Kowariancja i metody statystyczne",
    "wyklady/05-statystyczne-metody-uczenia-maszynowego.pdf":
        "Statystyczne metody uczenia maszynowego — redukcja wymiarowości i klasyfikacja",
    "wyklady/06-dwa-dni-z-uczeniem-maszynowym-komplet-slajdow.pdf":
        "Komplet slajdów całego szkolenia (wszystkie wykłady w jednym pliku)",
    "wyklady/07-podsumowanie-bloku-uczenia-maszynowego.pdf":
        "Wykład podsumowujący blok uczenia maszynowego",
    "wyklady/08-wyklad-wprowadzajacy-plan.pdf": "Wykład wprowadzający — plan i zakres",
    "wyklady/01-kwantowy-perceptron-komplet-materialow.pdf":
        "Kwantowy perceptron — komplet materiałów szkolenia w jednym pliku",
    "wyklady/02-slajdy-pomocnicze.pdf": "Slajdy pomocnicze do ćwiczeń",
    "wyklady/01-kwantowe-wyzarzanie-komplet-materialow.pdf":
        "Kwantowe wyżarzanie — komplet materiałów szkolenia w jednym pliku",
    "wyklady/01-kwantowe-uczenie-maszynowe-komplet-materialow.pdf":
        "Kwantowe uczenie maszynowe — komplet materiałów szkolenia w jednym pliku",
    "wyklady/02-kwantowe-uczenie-maszynowe-slajdy.pdf":
        "Kwantowe uczenie maszynowe i metody jądrowe — slajdy wykładowe",
    "warsztaty/pennylane-jax.ipynb":
        "PennyLane + JAX — obwód parametryzowalny i gradient metodą sprzężoną",
    "warsztaty/pennylane-jaxnn.ipynb":
        "PennyLane + JAX — trenowanie sieci neuronowej metodą spadku gradientu",
    "warsztaty/pennylane-mixed.ipynb":
        "PennyLane — obwody z szumem (symulator stanów mieszanych)",
    "materialy-zrodlowe/1811.02266-artykul-quantum-perceptron.pdf":
        "F. Tacchino i in., „An Artificial Neuron Implemented on an Actual Quantum "
        "Processor” (arXiv:1811.02266) — artykuł źródłowy",
}

# Opisy katalogów ćwiczeń (klucz: ścieżka względem katalogu bloku).
OPISY_KATALOGOW = {
    "warsztaty/d1e1z1-start": "Dzień 1, ćwiczenie 1 — wczytanie i oględziny danych",
    "warsztaty/d1e1z2-clustering": "Dzień 1, ćwiczenie 2 — klasteryzacja na danych modelowych",
    "warsztaty/d1e1z2-clustering-part-2": "Dzień 1, ćwiczenie 2 (cz. 2) — klasteryzacja danych rzeczywistych",
    "warsztaty/d1e1z3-anomalies": "Dzień 1, ćwiczenie 3 — detekcja anomalii",
    "warsztaty/d1e1z4-dimensionality": "Dzień 1, ćwiczenie 4 — redukcja wymiarowości (PCA, t-SNE)",
    "warsztaty/d1e1z5-end": "Dzień 1, ćwiczenie 5 — zadanie podsumowujące",
    "warsztaty/d1e2z1-zadanie": "Dzień 1, blok 2, ćwiczenie 1 — klasyfikacja",
    "warsztaty/d1e2z2-shap-svm": "Dzień 1, blok 2, ćwiczenie 2 — SVM i wyjaśnialność (SHAP)",
    "warsztaty/d1e2z3-cv": "Dzień 1, blok 2, ćwiczenie 3 — walidacja krzyżowa i miary jakości",
    "warsztaty/d2e3z1-start": "Dzień 2, ćwiczenie 1 — dane hiperspektralne Pavia University",
    "warsztaty/d2e3z2-neurons": "Dzień 2, ćwiczenie 2 — pojedynczy neuron",
    "warsztaty/d2e3z3-mlp": "Dzień 2, ćwiczenie 3 — sieć wielowarstwowa",
    "warsztaty/d2e3z4-mlp-torch": "Dzień 2, ćwiczenie 4 — MLP w PyTorch",
    "warsztaty/d2e4z1-conv": "Dzień 2, blok 4, ćwiczenie 1 — sploty i filtry",
    "warsztaty/d2e4z2-conv-networks": "Dzień 2, blok 4, ćwiczenie 2 — sieci konwolucyjne",
    "warsztaty/d2e4z3-llm": "Dzień 2, blok 4, ćwiczenie 3 — duże modele językowe",
    "warsztaty/dzien-1": "Dzień 1 — teoria: model Isinga, QUBO, twierdzenie adiabatyczne, D-Wave",
    "warsztaty/dzien-2": "Dzień 2 — praktyka: symulowane wyżarzanie, symulowana bifurkacja, GPU",
    "warsztaty/dzien-2/benchmarks": "Pomiary wydajności algorytmów (wymagają dodatkowego "
        "środowiska: GPU, CuPy/PyTorch CUDA)",
    "warsztaty/pliki_pomocnicze": "Moduły pomocnicze, instancje testowe i wyprowadzenia rachunkowe",
    "warsztaty": "Notatniki Jupyter z ćwiczeniami",
    "wyklady": "Slajdy i skrypty wykładowe w formacie PDF",
    "kod": "Przykładowe programy w Pythonie",
    "plan": "Program szkolenia",
    "zrodla-latex": "Źródła LaTeX slajdów (dla osób, które chcą je modyfikować)",
    "materialy-zrodlowe": "Publikacje naukowe, na których oparto ćwiczenia",
}
