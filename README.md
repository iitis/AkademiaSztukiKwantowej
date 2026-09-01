# Akademia Sztuki Kwantowej

Materiały dydaktyczne projektu **Akademia Sztuki Kwantowej**, realizowanego przez
Instytut Informatyki Teoretycznej i Stosowanej Polskiej Akademii Nauk w Gliwicach.
Wszystkie materiały są bezpłatne i dostępne bez logowania.

> **Najwygodniej korzystać ze strony z materiałami:**
> **[iitis.github.io/AkademiaSztukiKwantowej](https://iitis.github.io/AkademiaSztukiKwantowej/)**
> — slajdy otwierają się w przeglądarce, a notatniki Jupyter można przeczytać
> razem z wynikami obliczeń bez instalowania czegokolwiek.

## Materiały

Wszystkie materiały znajdują się w katalogu [`materialy/`](materialy/), w podziale
na pięć bloków tematycznych:

| Blok | Zawartość | Prowadzenie |
|------|-----------|-------------|
| [1. Obliczenia i algorytmy kwantowe](materialy/01-obliczenia-i-algorytmy-kwantowe/) | 6 plików PDF ze slajdami i skryptem (609 stron), przykłady w Pythonie (PennyLane) | P. Gawron, Z. Puchała, H. Wojewódka-Ściążko |
| [2. Dwa dni z uczeniem maszynowym](materialy/02-uczenie-maszynowe/) | 8 plików PDF ze slajdami (970 stron), 26 notatników Jupyter z ćwiczeniami i danymi | P. Głomb |
| [3. Kwantowy perceptron](materialy/03-kwantowy-perceptron/) | 2 pliki PDF ze slajdami i artykuł źródłowy (168 stron), 12 notatników Jupyter (Qiskit, IBM Quantum) | Ł. Pawela |
| [4. Kwantowe wyżarzanie](materialy/04-kwantowe-wyzarzanie/) | 1 plik PDF (136 stron), 24 notatniki Jupyter, moduły Python i instancje testowe | B. Gardas |
| [5. Kwantowe uczenie maszynowe i metody jądrowe](materialy/05-kwantowe-uczenie-maszynowe/) | 2 pliki PDF ze slajdami (225 stron), 3 notatniki Jupyter (PennyLane + JAX), przykłady w Pythonie, źródła LaTeX | P. Gawron |

## Jak korzystać z materiałów

**Chcę tylko przeczytać wykład.** Otwórz
[stronę z materiałami](https://iitis.github.io/AkademiaSztukiKwantowej/) i kliknij
tytuł wykładu — plik PDF otworzy się w przeglądarce. Te same pliki znajdują się
w katalogu `materialy/<blok>/wyklady/`.

**Chcę zobaczyć ćwiczenia razem z wynikami.** Na stronie każdego bloku, w tabeli
notatników, kliknij *„czytaj w przeglądarce”*. Zobaczysz kod, wykresy i wyniki
obliczeń bez instalowania Pythona.

**Chcę samodzielnie uruchomić kod.** Pobierz materiały i zainstaluj wymagane pakiety:

```bash
git clone https://github.com/iitis/AkademiaSztukiKwantowej.git
cd AkademiaSztukiKwantowej/materialy/04-kwantowe-wyzarzanie   # przykładowy blok
pip install -r warsztaty/requirements.txt
jupyter notebook
```

Jeżeli nie korzystasz z gita, użyj przycisku **Code → Download ZIP** na górze tej
strony — pobierzesz całe repozytorium jako jedno archiwum.

## Nagrania wykładów

Nagrania wideo wykładów online (40 nagrań, ok. 15 GB) są dostępne:

- w systemie [akademia.iitis.pl](https://akademia.iitis.pl/) — przy poszczególnych wydarzeniach,
- w repozytorium Zenodo, DOI [10.5281/zenodo.22231106](https://doi.org/10.5281/zenodo.22231106),
- archiwum wszystkich wydarzeń wraz z programami: [ask.iitis.pl/archiwum](https://ask.iitis.pl/archiwum/).

## Struktura repozytorium

```
materialy/     komplet materiałów dydaktycznych (slajdy PDF, notatniki, kod, dane)
strona/        źródła strony z materiałami (GitHub Pages) i podglądy notatników
narzedzia/     skrypt generujący stronę i pliki README
index.html     strona główna serwisu z materiałami
```

Pliki robocze, z których powstały materiały (źródła LaTeX wraz z rysunkami, wersje
robocze notatników, pełna historia zmian), pozostają w gałęziach autorskich:
[`pg/qcintro`](../../tree/pg/qcintro),
[`pg_final`](../../tree/pg_final),
[`lp/perceptron`](../../tree/lp/perceptron),
[`bg/quantum_annealing`](../../tree/bg/quantum_annealing),
[`pg/szkolenie`](../../tree/pg/szkolenie),
[`mlintro`](../../tree/mlintro),
[`zp/QI_intro_lectures`](../../tree/zp/QI_intro_lectures).
Do korzystania z materiałów nie jest potrzebny dostęp do tych gałęzi.

## Jak cytować

> Materiały Akademii Sztuki Kwantowej, Instytut Informatyki Teoretycznej i Stosowanej
> Polskiej Akademii Nauk, Gliwice 2024–2026,
> https://github.com/iitis/AkademiaSztukiKwantowej

## Licencja

Apache 2.0 — szczegóły w pliku [LICENSE](LICENSE).

## Finansowanie

*Projekt dofinansowany ze środków budżetu państwa, przyznanych przez Ministra Edukacji
i Nauki w ramach Programu „Nauka dla Społeczeństwa II”*

Numer projektu: **NdS-II/SP/0222/2024/01** · Termin realizacji: **03/04/2024 – 03/04/2026** ·
Kwota przyznana: **1 000 000 PLN** · Źródło finansowania: **Ministerstwo Nauki i Szkolnictwa Wyższego**

![Ministerstwo Nauki i Szkolnictwa Wyższego](https://ask.iitis.pl/assets/images/logo1.png) ![Nauka Dla Społeczeństwa](https://ask.iitis.pl/assets/images/logo2.png)
