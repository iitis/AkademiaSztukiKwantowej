import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['image.interpolation'] = 'nearest'

rec = np.load('PaviaU.npz')
data = rec['data']
X = np.reshape(data, (-1, data.shape[-1]))
ground_truth = rec['gt']
y = ground_truth.ravel()


clf = IsolationForest(contamination=0.01)
anomalies = clf.fit_predict(X)
anomalies2 = np.reshape(anomalies, ground_truth.shape)
plt.imshow(anomalies2, cmap=plt.cm.bwr_r)
plt.show()

data2 = np.delete(data, (142), axis=0)
ground_truth2 = np.delete(ground_truth, (142), axis=0)
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(data[:, :, 50])
plt.subplot(122)
plt.imshow(data2[:, :, 50])
plt.show()
X = np.reshape(data2, (-1, data.shape[-1]))
y = ground_truth2.ravel()

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
