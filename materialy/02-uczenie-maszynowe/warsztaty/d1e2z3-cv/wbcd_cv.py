import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
param_grid = {
    "svc__C":     10.0 ** np.arange(-2, 4),   # 0.01 … 1 000
    "svc__gamma": 10.0 ** np.arange(-4, 2),   # 0.0001 … 1
}
inner = StratifiedKFold(n_splits=5, shuffle=True)
outer = StratifiedKFold(n_splits=5, shuffle=True)

grid = GridSearchCV(pipe, param_grid, cv=inner)
scores = cross_val_score(grid, X, y, cv=outer)

print(f"Nested 5×5 CV accuracy: {scores.mean():.3f} ± {scores.std(ddof=1):.3f}")
