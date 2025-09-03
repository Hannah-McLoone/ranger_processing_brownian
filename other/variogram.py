import numpy as np
import matplotlib.pyplot as plt
import skgstat as skg
from numpy.fft import fft2, ifft2, fftshift
from scipy.spatial.distance import pdist, squareform
import sys 

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <park>")
        sys.exit(1)

    park = sys.argv[1]
    values = np.loadtxt(f"{park}_speed90.csv", delimiter=",")
    intensity = np.loadtxt(f"{park}_intensity90.csv", delimiter=",")
    values[intensity < 100] = np.nan


    height, width = values.shape
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    coords = np.column_stack([x_coords.flatten(), y_coords.flatten()])
    values = values.ravel()



    mask = ~np.isnan(values)
    coords = coords[mask]
    values = values[mask]


    V = skg.Variogram(coords, values, n_lags=100)  # default is usually 6-10
    np.savetxt(f"{park}_variogram.csv",np.array(V.experimental), delimiter=",", fmt='%.6f')
    np.savetxt(f"{park}_variogram_freq.csv",np.array(V.bin_count ), delimiter=",", fmt='%.6f') # range(len(my_data))
    np.savetxt(f"{park}_variogram_bins.csv",np.array(V.bins), delimiter=",", fmt='%.6f')