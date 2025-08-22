from aligning import create_feature_speed_intesity_tables
import numpy as np
import scanpy as sc
import anndata
import random
import pandas as pd

def make_block_cv_ids(height, width, block_size, n_folds=5):
    # number of blocks along each dimension
    n_blocks_y = (height + block_size - 1) // block_size
    n_blocks_x = (width + block_size - 1) // block_size

    # assign IDs to blocks in a meshgrid
    block_ids = np.arange(n_blocks_y * n_blocks_x).reshape(n_blocks_y, n_blocks_x)
    block_ids = block_ids % n_folds + 1  # cycle through 1..n_folds

    # expand block IDs into pixel grid
    grid = np.kron(block_ids, np.ones((block_size, block_size), dtype=int))

    # crop to exact height/width
    grid = grid[:height, :width]
    return grid


areas = ['oban']
feature_list = ['ndvi','biomass','cover','elevation', 'slope']
resolution = np.float64(0.0001 * 10 * 10) 



width = int(round(x[1] - x[0],6) / resolution)# + 1
height =  int(round(y[1] - y[0],6) / resolution)# + 1 #check these!!!!!!!
df = create_feature_speed_intesity_tables(resolution, areas, feature_list, x, y)


features = pd.DataFrame({'speed': df['speed']})
features = features.fillna(0)#.iloc[:, :-2].values

#cv_grid = make_block_cv_ids(height, width, block_size = 4, n_folds = 5)


x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
coords = np.column_stack([x_coords.flatten(), y_coords.flatten()])




adata = anndata.AnnData(X=features)
adata.obsm['spatial'] = coords  # store coordinates

sc.pp.neighbors(adata, use_rep='spatial', n_neighbors=8, knn=True)
x = sc.metrics.morans_i(adata)
print(x)

vals = []
for n in range (1,100):
    non_zero_idx = features['speed'] != 0

    # Shuffle only non-zero values
    features.loc[non_zero_idx, 'speed'] = np.random.permutation(features.loc[non_zero_idx, 'speed'].values)

    adata = anndata.AnnData(X=features)
    adata.obsm['spatial'] = coords  # store coordinates
    sc.pp.neighbors(adata, use_rep='spatial', n_neighbors=8, knn=True)
    mi = sc.metrics.morans_i(adata)
    vals.append(mi)


lower = np.percentile(vals, 5)
upper = np.percentile(vals, 95)

# Check if x is in the outer 5%
if x < lower or x > upper:
    print(f"significant")
else:
    print(":)")