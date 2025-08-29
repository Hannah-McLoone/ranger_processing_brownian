import numpy as np
import matplotlib.pyplot as plt
import skgstat as skg
from numpy.fft import fft2, ifft2, fftshift
from scipy.spatial.distance import pdist, squareform

values = np.loadtxt(f"variogram.csv", delimiter=",")
x = [i * 12.89 for i in range (0,100)]
plt.plot(x[0:60], values[0:int(len(values) * 0.6)])
plt.xlabel('pixel seperation')
plt.ylabel('semivariance')
plt.show()

values = np.loadtxt(f"variogram_freq.csv", delimiter=",")
plt.plot(x, values)
plt.xlabel('pixel seperation')
plt.ylabel('frequency')
plt.show()



values = np.loadtxt(f"oban_speed90.csv", delimiter=",")
intensity = np.loadtxt(f"oban_intensity90.csv", delimiter=",")
values[intensity < 100] = np.nan
print(len(values))
values = values[::5,::5]

height, width = values.shape
x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
coords = np.column_stack([x_coords.flatten(), y_coords.flatten()])
values = values.ravel()



mask = ~np.isnan(values)
coords = coords[mask]
values = values[mask]


V = skg.Variogram(coords, values, n_lags=100)  # default is usually 6-10
V.plot()
plt.show()
#np.savetxt("variogram.csv",np.array(V.experimental), delimiter=",", fmt='%.6f')
#np.savetxt("variogram_freq.csv",np.array(V.bin_count ), delimiter=",", fmt='%.6f')# range(len(my_data))