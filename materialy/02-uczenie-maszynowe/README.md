# 2. Dwa dni z uczeniem maszynowym

Wprowadzenie do klasycznego (nie-kwantowego) uczenia maszynowego: analiza danych, klasteryzacja, redukcja wymiarowości, detekcja anomalii, maszyny wektorów nośnych, sieci neuronowe i konwolucyjne, duże modele językowe. Blok stanowi podstawę do metod kwantowych.

**Prowadzenie:** dr hab. inż. Przemysław Głomb

Wygodniejszy w przeglądaniu spis tych materiałów, z podglądem notatników w przeglądarce, znajduje się na stronie [iitis.github.io/AkademiaSztukiKwantowej](https://iitis.github.io/AkademiaSztukiKwantowej/strona/blok-02.html).

## Slajdy i skrypty (PDF)

| Materiał | Plik | Stron |
|---|---|---|
| Perceptron | [`wyklady/01-perceptron.pdf`](wyklady/01-perceptron.pdf) | 60 |
| Sieci wielowarstwowe (MLP) | [`wyklady/02-sieci-mlp.pdf`](wyklady/02-sieci-mlp.pdf) | 110 |
| Sieci MLP — analiza działania | [`wyklady/03-sieci-mlp-analiza.pdf`](wyklady/03-sieci-mlp-analiza.pdf) | 34 |
| Kowariancja i metody statystyczne | [`wyklady/04-kowariancja-i-metody-statystyczne.pdf`](wyklady/04-kowariancja-i-metody-statystyczne.pdf) | 94 |
| Statystyczne metody uczenia maszynowego — redukcja wymiarowości i klasyfikacja | [`wyklady/05-statystyczne-metody-uczenia-maszynowego.pdf`](wyklady/05-statystyczne-metody-uczenia-maszynowego.pdf) | 172 |
| Komplet slajdów całego szkolenia (wszystkie wykłady w jednym pliku) | [`wyklady/06-dwa-dni-z-uczeniem-maszynowym-komplet-slajdow.pdf`](wyklady/06-dwa-dni-z-uczeniem-maszynowym-komplet-slajdow.pdf) | 484 |
| Wykład podsumowujący blok uczenia maszynowego | [`wyklady/07-podsumowanie-bloku-uczenia-maszynowego.pdf`](wyklady/07-podsumowanie-bloku-uczenia-maszynowego.pdf) | 14 |
| Wykład wprowadzający — plan i zakres | [`wyklady/08-wyklad-wprowadzajacy-plan.pdf`](wyklady/08-wyklad-wprowadzajacy-plan.pdf) | 2 |

## Notatniki Jupyter

Notatniki można przeczytać w przeglądarce (bez instalacji) korzystając z podglądów na stronie projektu, albo pobrać i uruchomić lokalnie.

| Notatnik | Zawartość |
|---|---|
| [`warsztaty/d1e1z1-start/start.ipynb`](warsztaty/d1e1z1-start/start.ipynb) | Start |
| [`warsztaty/d1e1z1-start/start_inspect.ipynb`](warsztaty/d1e1z1-start/start_inspect.ipynb) | Start inspect |
| [`warsztaty/d1e1z2-clustering/cluster_toy.ipynb`](warsztaty/d1e1z2-clustering/cluster_toy.ipynb) | Cluster toy |
| [`warsztaty/d1e1z2-clustering-part-2/cluster_real.ipynb`](warsztaty/d1e1z2-clustering-part-2/cluster_real.ipynb) | Cluster real |
| [`warsztaty/d1e1z2-clustering-part-2/cluster_real_demo.ipynb`](warsztaty/d1e1z2-clustering-part-2/cluster_real_demo.ipynb) | Cluster real demo |
| [`warsztaty/d1e1z3-anomalies/anomalies.ipynb`](warsztaty/d1e1z3-anomalies/anomalies.ipynb) | Anomalies |
| [`warsztaty/d1e1z3-anomalies/anomalies_cov.ipynb`](warsztaty/d1e1z3-anomalies/anomalies_cov.ipynb) | Anomalies cov |
| [`warsztaty/d1e1z4-dimensionality/faces.ipynb`](warsztaty/d1e1z4-dimensionality/faces.ipynb) | Faces |
| [`warsztaty/d1e1z4-dimensionality/faces_full.ipynb`](warsztaty/d1e1z4-dimensionality/faces_full.ipynb) | Faces full |
| [`warsztaty/d1e1z4-dimensionality/pca_toy.ipynb`](warsztaty/d1e1z4-dimensionality/pca_toy.ipynb) | Pca toy |
| [`warsztaty/d1e1z4-dimensionality/pca_tsne.ipynb`](warsztaty/d1e1z4-dimensionality/pca_tsne.ipynb) | Pca tsne |
| [`warsztaty/d1e1z5-end/blood.ipynb`](warsztaty/d1e1z5-end/blood.ipynb) | Blood |
| [`warsztaty/d1e2z1-zadanie/classify.ipynb`](warsztaty/d1e2z1-zadanie/classify.ipynb) | Classify |
| [`warsztaty/d1e2z2-shap-svm/wbcd_shap.ipynb`](warsztaty/d1e2z2-shap-svm/wbcd_shap.ipynb) | Wbcd shap |
| [`warsztaty/d1e2z2-shap-svm/wbcd_svm.ipynb`](warsztaty/d1e2z2-shap-svm/wbcd_svm.ipynb) | Wbcd svm |
| [`warsztaty/d1e2z3-cv/measures_cv.ipynb`](warsztaty/d1e2z3-cv/measures_cv.ipynb) | Measures cv |
| [`warsztaty/d1e2z3-cv/wbcd_cv.ipynb`](warsztaty/d1e2z3-cv/wbcd_cv.ipynb) | Wbcd cv |
| [`warsztaty/d2e3z2-neurons/neuron.ipynb`](warsztaty/d2e3z2-neurons/neuron.ipynb) | Neuron |
| [`warsztaty/d2e3z3-mlp/mlp_ex.ipynb`](warsztaty/d2e3z3-mlp/mlp_ex.ipynb) | Mlp ex |
| [`warsztaty/d2e3z4-mlp-torch/frameworks.ipynb`](warsztaty/d2e3z4-mlp-torch/frameworks.ipynb) | Frameworks |
| [`warsztaty/d2e3z4-mlp-torch/pytmlp.ipynb`](warsztaty/d2e3z4-mlp-torch/pytmlp.ipynb) | Pytmlp |
| [`warsztaty/d2e3z4-mlp-torch/pytmlp_val.ipynb`](warsztaty/d2e3z4-mlp-torch/pytmlp_val.ipynb) | Pytmlp val |
| [`warsztaty/d2e4z1-conv/conv2.ipynb`](warsztaty/d2e4z1-conv/conv2.ipynb) | Conv2 |
| [`warsztaty/d2e4z2-conv-networks/alex.ipynb`](warsztaty/d2e4z2-conv-networks/alex.ipynb) | Alex |
| [`warsztaty/d2e4z3-llm/llm_tinyllama.ipynb`](warsztaty/d2e4z3-llm/llm_tinyllama.ipynb) | Llm tinyllama |
| [`warsztaty/d2e4z3-llm/tinyllama_new.ipynb`](warsztaty/d2e4z3-llm/tinyllama_new.ipynb) | Tinyllama new |

## Jak uruchomić ćwiczenia

```bash
git clone https://github.com/iitis/AkademiaSztukiKwantowej.git
cd AkademiaSztukiKwantowej/materialy/02-uczenie-maszynowe
pip install -r warsztaty/requirements.txt
jupyter notebook
```

## Materiały źródłowe

Pliki robocze (źródła LaTeX z rysunkami, wersje robocze, historia zmian) pozostają w gałęziach autorskich: [`pg_final`](https://github.com/iitis/AkademiaSztukiKwantowej/tree/pg_final), [`mlintro`](https://github.com/iitis/AkademiaSztukiKwantowej/tree/mlintro).

## Licencja

Apache 2.0 — patrz plik [LICENSE](../../LICENSE) w katalogu głównym repozytorium.
