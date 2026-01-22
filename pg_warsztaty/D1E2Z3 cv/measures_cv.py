# Ocena wydajności klasyfikatorów
# Porównanie dwóch klasyfikatorów dla określonego zbioru danych: problem doboru hiperparametrów, wyboru miary, oceny istotności statystycznej
from pprint import pprint

import numpy as np, pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import shapiro, ttest_rel, wilcoxon


# Wczytanie (tu: wylosowanie) zbioru danych. Zbiór niezbalansowany, znaczna przewaga jednej klasy
X, y = make_classification(
    n_samples=2000, n_features=20, n_informative=4, n_redundant=2, n_repeated=0,
    n_clusters_per_class=2, weights=[0.97, 0.03], flip_y=0.0, class_sep=1.0,
    random_state=42
)


# Przygotowanie zmiennych: walidacja krzyżowa, parametry SVM itd
outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
param_grid = {"svc__C": [0.3,1,3,10], 
              "svc__gamma": ["scale", 0.03, 0.1]}
metrics = ["accuracy","balanced_accuracy","precision","recall","f1"]
svm_scores = {m:[] for m in metrics}
knn_scores = {m:[] for m in metrics}
p_value_T = 0.05


# Pętla główna - przejscie przez foldy walidacji, trenowanie modeli, wyznaczenie miar
for tr, te in outer.split(X,y):
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]

    svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=42))
    gs = GridSearchCV(svm, param_grid, cv=inner, scoring="balanced_accuracy", n_jobs=1)
    gs.fit(Xtr, ytr)
    yhat_svm = gs.predict(Xte)

    knn = KNeighborsClassifier(n_neighbors=1).fit(Xtr, ytr)
    yhat_knn = knn.predict(Xte)

    for name, func in [
        ("accuracy", accuracy_score),
        ("balanced_accuracy", balanced_accuracy_score),
        ("precision", lambda yt, yp: precision_score(yt, yp, zero_division=0)),
        ("recall",    lambda yt, yp: recall_score(yt, yp, zero_division=0)),
        ("f1",        lambda yt, yp: f1_score(yt, yp, zero_division=0)),
    ]:
        svm_scores[name].append(func(yte, yhat_svm))
        knn_scores[name].append(func(yte, yhat_knn))
pprint(svm_scores)


# Ocena statystyczna
rows, tests = [], []
for m in metrics:
    s = np.array(svm_scores[m])
    k = np.array(knn_scores[m])
    d = s - k
    W, p_norm = shapiro(d)
    if p_norm >= p_value_T:
        stat, p = ttest_rel(s, k)
        test_used = "paired t-test"
    else:
        stat, p = wilcoxon(s, k)
        test_used = "Wilcoxon signed-rank"
    rows.append([m, s.mean(), s.std(ddof=1), k.mean(), k.std(ddof=1)])
    tests.append([m, test_used, float(p_norm), float(stat), float(p), "YES" if p < p_value_T else "NO"])
    

# Przygotowanie i wyswietlenie tabel wynikowych
summary = pd.DataFrame(rows, columns=["metric","svm_mean","svm_sd","knn1_mean","knn1_sd"]).round(4)
tests_df = pd.DataFrame(tests, columns=["metric","test","shapiro_p","stat","p_value",f"significant(p<{p_value_T:.2f})"]).round(4)
print(summary)
print(tests_df)
