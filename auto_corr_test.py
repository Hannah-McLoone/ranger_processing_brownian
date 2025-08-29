import numpy as np
import scanpy as sc
import anndata
import random
import pandas as pd


def downsample_2d(arr, factor=10):
    # Get new shape
    h, w = arr.shape
    new_h, new_w = h // factor, w // factor
    
    # Trim array so it's divisible by factor
    arr = arr[:new_h*factor, :new_w*factor]
    
    # Reshape into blocks
    reshaped = arr.reshape(new_h, factor, new_w, factor)
    
    # Take mean over the 10x10 blocks (ignoring NaNs)
    downsampled = np.nanmean(np.nanmean(reshaped, axis=3), axis=1)
    
    return downsampled

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

morans_i_values = []
significant = []
for n in range (2**6,2**8,10):
    speed = np.loadtxt('oban_speed90.csv', delimiter=",")

    speed = downsample_2d(speed, n)

    speed[np.isnan(speed)] = 0
    height, width = speed.shape
    features = speed.flatten()[:, None]

    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    coords = np.column_stack([x_coords.flatten(), y_coords.flatten()])

    adata = anndata.AnnData(X=features)
    adata.obsm['spatial'] = coords  # store coordinates
    sc.pp.neighbors(adata, use_rep='spatial', n_neighbors=8, knn=True)
    x = sc.metrics.morans_i(adata)
    morans_i_values.append(x)




    vals = []
    if False:# for i in range (1,1):
        non_zero_idx = features != 0

        # Shuffle only non-zero values
        features[non_zero_idx] = np.random.permutation(features[non_zero_idx])

        adata = anndata.AnnData(X=features)
        adata.obsm['spatial'] = coords  # store coordinates
        sc.pp.neighbors(adata, use_rep='spatial', n_neighbors=8, knn=True)
        mi = sc.metrics.morans_i(adata)
        vals.append(mi)


    #lower = np.percentile(vals, 5)
    #upper = np.percentile(vals, 95)

    # Check if x is in the outer 5%
    #if x < lower or x > upper:
    #    significant.append(1)
    #else:
    #    significant.append(0)


np.savetxt(f"morans_i_full.csv", np.array(morans_i_values), delimiter=",", fmt='%.6f')
#print(morans_i_values)
#np.savetxt(f"significance.csv", np.array(significant), delimiter=",", fmt='%.6f')