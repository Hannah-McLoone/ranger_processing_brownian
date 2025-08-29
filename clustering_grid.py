import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, SpectralClustering, OPTICS
from sklearn.neighbors import NearestNeighbors
from matplotlib import colors
from scipy.stats import rankdata
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering

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





arr = np.loadtxt('output/intensity.csv', delimiter=",")
arr = arr[::-1]
arr = arr[3000:4000, 2000:3500]   # first 100 rows & columns
arr[arr<100] = 0


arr = rankdata(arr, method='average').reshape(arr.shape) / (arr.shape[0] * arr.shape[1])
arr[arr<0.8] = 0
arr[arr > 0] = (arr[arr > 0] - 0.8) * 5
arr = arr * 10

# Cluster with DBSCAN



plt.imshow(arr, cmap="hot", interpolation="nearest")# gist_ncar
plt.colorbar(label="Cluster ID")
plt.title("DBSCAN Clusters on 2D Grid")
plt.show()

coords = np.argwhere(arr != 0)
#mask = np.random.rand(len(coords)) < np.array([arr[tuple(c)] for c in coords])
#coords = coords[mask]

#coords_with_values = np.array([[i, j, arr[i, j]] for i, j in coords])


knn_graph = NearestNeighbors(n_neighbors=30, algorithm='ball_tree', n_jobs=-1).fit(coords)
distances, indices = knn_graph.kneighbors(coords)

# Use the graph to cluster
clustering = AgglomerativeClustering(
    n_clusters=None,       # or set a number if you know how many clusters
    distance_threshold=25000, # play with this to control cluster formation
    connectivity=knn_graph.kneighbors_graph(coords)
).fit(coords)

labels = clustering.labels_

print('!')
coords = coords[np.isin(labels, [1])]
labels = labels[np.isin(labels, [1])]

#__________________________________________________________
# Cluster with DBSCAN

# Multiply instances of each coord by its value in arr
expanded_coords = []
for (i, j) in coords:
    count = int(arr[i, j])   # number of times to repeat
    expanded_coords.extend([[i, j]] * count)

coords_expanded = np.array(expanded_coords)


db = DBSCAN(eps=5, min_samples=20).fit(coords_expanded)

# Map back to original coords
labels_expanded = db.labels_

# Since coords_expanded has duplicates, compress back to unique coords
sub_labels = np.full(len(coords), -1)
for idx, (i, j) in enumerate(coords):
    # Take the most common label among its duplicates
    dup_labels = labels_expanded[np.all(coords_expanded == [i, j], axis=1)]
    if len(dup_labels) > 0:
        # exclude noise (-1) if possible
        vals, counts = np.unique(dup_labels, return_counts=True)
        if np.any(vals != -1):
            vals, counts = zip(*[(v, c) for v, c in zip(vals, counts) if v != -1])
        sub_labels[idx] = vals[np.argmax(counts)]

#_______________________________________________________________________





# Create a grid for visualization
cluster_grid = np.full(arr.shape, np.nan)
for (r, c), label in zip(coords, sub_labels):
    cluster_grid[r, c] = label

# Plot
plt.imshow(cluster_grid, cmap="prism", interpolation="nearest")# gist_ncar
plt.colorbar(label="Cluster ID")
plt.title("DBSCAN Clusters on 2D Grid")
plt.show()




arr = rankdata(arr, method='average').reshape(arr.shape) / (arr.shape[0] * arr.shape[1])
arr[arr<0.8] = 0
arr[arr > 0] = (arr[arr > 0] - 0.8) * 5
arr = arr *3
#arr = arr * 1500000