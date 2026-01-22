import matplotlib.pyplot as plt
import numpy as np

"""

Wczytaj obrazek do analizy. d01_frame_I300 jest łatwiejszy, możesz 
odkomentować d01_comparison_I350 jeżeli chcesz trudniejszy :)

"""
    

fname = 'd01_frame_I300'
fname = 'd01_comparison_I350'
record = np.load(f'{fname}.npz')


data, rgb, wavelengths = [record[k] for k in ['data', 'rgb', 'wavelengths']]

i, j = 160, 160 # Example point

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(rgb)
plt.scatter([i], [j], c='w', s=100, alpha=0.5)
plt.scatter([i], [j], c='k', s=10, alpha=0.5)
plt.subplot(122)
plt.plot(wavelengths, data[i, j], 'k-')
plt.ylabel('Reflektancja')
plt.xlabel('Długosc swiatła [nm]')
plt.tight_layout()
plt.show()
