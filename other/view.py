

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import rankdata

# Load CSV file as NumPy array
csv_file = "intensity2.csv"  # Replace with your CSV file path
data = np.loadtxt(csv_file, delimiter=",")  # Assumes all numeric

# Get shape & size
shape = data.shape
size = shape[0] * shape[1]

# Rank-normalize
#ranks = rankdata(data, method='max').reshape(shape) / size

# Create heatmap
#ranks[ranks>0.]
data[data<20] = 0
#data = rankdata(data, method='max').reshape(shape) / size





csv_file = "speed2.csv"  # Replace with your CSV file path
speed = np.loadtxt(csv_file, delimiter=",")  # Assumes all numeric

result = np.where(data != 0, speed, np.nan)


plt.imshow(result, cmap='hot',origin='lower')
plt.colorbar()
plt.show()