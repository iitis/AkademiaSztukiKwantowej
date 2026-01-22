# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def setup_fig():
    plt.figure(figsize=(10, 3))
    mpl.rcParams.update({'font.size': 12})
    ticks = [400, 542, 576, 950, 1000]
    for b in [542, 576, 950]:
        plt.axvline(x=b, color='k', linestyle='--')
    plt.xticks(ticks, rotation=40)
    plt.xlabel('Długość światła [nm]', labelpad=-10)
    plt.tight_layout()
   

def plot_spectra(bands, data, absorbance=False, c='r', **kwargs):
    s = np.log(1.0 / data) if absorbance else data
    plt.ylabel('Absorbancja' if absorbance else 'Reflektancja')
    plt.plot(bands, s, c=c, **kwargs)  


X = np.load('blood1X.npz')['X']
bands = np.load('blood_bands.npy')
setup_fig()
plot_spectra(bands, X[28], absorbance=False, c='r')
plot_spectra(bands, X[37], absorbance=False, c='r')
plot_spectra(bands, X[107], absorbance=False, c='r')

plot_spectra(bands, X[0], absorbance=False, c='b')
plot_spectra(bands, X[50], absorbance=False, c='b')
plot_spectra(bands, X[100], absorbance=False, c='b')    
plt.show()
