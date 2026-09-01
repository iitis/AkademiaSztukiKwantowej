import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

# 1) data  ──────────────────────────────────────────────────────────
X, y = load_digits(return_X_y=True)
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, 
                    test_size=.30, stratify=y, random_state=0)
X_val, X_te,  y_val, y_te = train_test_split(X_tmp, y_tmp, 
                    test_size=.50, stratify=y_tmp, random_state=0)

# 2) search space  ──────────────────────────────────────────────────
param_grid = dict(hidden_layer_sizes=[(128,), (256, 128)],
                  activation=["relu", "tanh"],
                  solver=["adam", "sgd"],
                  batch_size=[64, 128],
                  learning_rate_init=[1e-3, 1e-2])
print(param_grid)

# 3) brute-force grid search on validation set  ─────────────────────
best_pipe, best_val = None, -1
for p in ParameterGrid(param_grid):
    pipe = Pipeline([("sc", StandardScaler()),
                     ("mlp", MLPClassifier(max_iter=50, random_state=0, **p))])
    pipe.fit(X_tr, y_tr)
    val_score = pipe.score(X_val, y_val)
    if val_score > best_val:
        best_val, best_pipe = val_score, pipe

# 4) report & plot  ─────────────────────────────────────────────────
print(f"validation accuracy : {best_val:.3f}")
print(f"test accuracy       : {best_pipe.score(X_te, y_te):.3f}")

plt.plot(best_pipe.named_steps["mlp"].loss_curve_, 'x-')
plt.title("Loss curve (best MLP)"); plt.xlabel("iteration"); plt.ylabel("loss")
plt.tight_layout(); plt.show()
