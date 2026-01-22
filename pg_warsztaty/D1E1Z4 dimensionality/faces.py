# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA

"""

Wczytanie zbioru twarzy Olivetti

"""

faces = fetch_olivetti_faces(shuffle=True, random_state=42)
X = faces.data # shape (400, 64*64)
dim = faces.images.shape[1] # 64

