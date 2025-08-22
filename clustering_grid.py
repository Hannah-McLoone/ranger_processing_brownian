import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from matplotlib import colors

arr = np.array([[0,0,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                 [0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                 [0,0,1,2,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                 [0,0,0.5,2,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                 [0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                 [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                 [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1.5,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0,0,1,1.5,2,2,2,2,2,2,1,1,1,0,0],
                 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])



plt.imshow(arr, cmap='hot')
plt.colorbar()
plt.show()

coords = np.argwhere(arr != 0)

# Cluster with DBSCAN
db = DBSCAN(eps=1.5, min_samples=2).fit(coords)
labels = db.labels_

# Create a grid for visualization
cluster_grid = np.full(arr.shape, -1)  # fill with -1 (noise)
for (r, c), label in zip(coords, labels):
    cluster_grid[r, c] = label

# Plot
plt.imshow(cluster_grid, cmap="tab20", interpolation="nearest")
plt.colorbar(label="Cluster ID")
plt.title("DBSCAN Clusters on 2D Grid")
plt.show()