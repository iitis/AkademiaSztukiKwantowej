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
