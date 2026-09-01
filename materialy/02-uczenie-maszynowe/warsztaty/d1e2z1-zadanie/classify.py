import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from sklearn.metrics import accuracy_score


def loader(fname):
    record = np.load(fname, allow_pickle=True)
    return record['X'], record['y']


def test(clf, fname):
    X_test, y_test = loader(fname) 
    y_predict = clf.predict(X_test)
    a = accuracy_score(y_test, y_predict)
    print(f'Skutecznosć = {a:.2%}')

"""

Wczytanie zbioru danych - klasyfikacja komórek z rakiem po biopsji.

Zobacz intro.jpg, pic.jpg

"""


X_train, y_train = loader('train.npz')

"""

Tu wstaw swój klasyfikator; możesz dowolnie rozwijać kod.

"""

clf = None#Classifier(parameter=value)
clf.fit(X_train, y_train)

test(clf, 'train.npz')
#test(clf, 'test.npz')
