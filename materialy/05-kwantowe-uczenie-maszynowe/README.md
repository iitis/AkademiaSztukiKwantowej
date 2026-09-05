# 5. Kwantowe uczenie maszynowe i kwantowe metody jądrowe

Rozwiązywanie problemów uczenia maszynowego za pomocą komputerów kwantowych: obwody parametryzowalne, algorytmy wariacyjne, kwantowe sieci neuronowe, kwantowe metody jądrowe, modele hybrydowe (PyTorch/JAX + PennyLane) oraz wpływ szumu na jakość modeli.

**Prowadzenie:** dr hab. inż. Piotr Gawron

Wygodniejszy w przeglądaniu spis tych materiałów, z podglądem notatników w przeglądarce, znajduje się na stronie [iitis.github.io/AkademiaSztukiKwantowej](https://iitis.github.io/AkademiaSztukiKwantowej/strona/blok-05.html).

## Slajdy i skrypty (PDF)

| Materiał | Plik | Stron |
|---|---|---|
| Kwantowe uczenie maszynowe — komplet materiałów szkolenia w jednym pliku | [`wyklady/01-kwantowe-uczenie-maszynowe-komplet-materialow.pdf`](wyklady/01-kwantowe-uczenie-maszynowe-komplet-materialow.pdf) | 114 |
| Kwantowe uczenie maszynowe i metody jądrowe — slajdy wykładowe | [`wyklady/02-kwantowe-uczenie-maszynowe-slajdy.pdf`](wyklady/02-kwantowe-uczenie-maszynowe-slajdy.pdf) | 111 |

## Notatniki Jupyter

Notatniki można przeczytać w przeglądarce (bez instalacji) korzystając z podglądów na stronie projektu, albo pobrać i uruchomić lokalnie.

| Notatnik | Zawartość |
|---|---|
| [`warsztaty/pennylane-jax.ipynb`](warsztaty/pennylane-jax.ipynb) | PennyLane + JAX — obwód parametryzowalny i gradient metodą sprzężoną |
| [`warsztaty/pennylane-jaxnn.ipynb`](warsztaty/pennylane-jaxnn.ipynb) | PennyLane + JAX — trenowanie sieci neuronowej metodą spadku gradientu |
| [`warsztaty/pennylane-mixed.ipynb`](warsztaty/pennylane-mixed.ipynb) | PennyLane — obwody z szumem (symulator stanów mieszanych) |

## Jak uruchomić ćwiczenia

```bash
git clone https://github.com/iitis/AkademiaSztukiKwantowej.git
cd AkademiaSztukiKwantowej/materialy/05-kwantowe-uczenie-maszynowe
pip install -r warsztaty/requirements.txt
jupyter notebook
```

## Materiały źródłowe

Pliki robocze (źródła LaTeX z rysunkami, wersje robocze, historia zmian) pozostają w gałęziach autorskich: [`pg/szkolenie`](https://github.com/iitis/AkademiaSztukiKwantowej/tree/pg/szkolenie).

## Licencja

Apache 2.0 — patrz plik [LICENSE](../../LICENSE) w katalogu głównym repozytorium.
