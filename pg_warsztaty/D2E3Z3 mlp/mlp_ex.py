import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

# load digits from sklearn datasets
# train_test_split to separate training, validation, test
# prepare pipieline with StandardScaler and MLPClassifier
# setup param_grid with hidden_layer_sizes [(128,), (256, 128)], activation ["relu", "tanh"], solver ["adam", "sgd"], batch_size [64, 128], learning_rate_init [1e-3, 1e-2]
# select hyperparameters based on accuracy on validation set
# for best model print validation and test accuracy, and loss curve 