from other.aligning import create_feature_speed_intesity_tables
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






rank = np.loadtxt('output/intensity2.csv', delimiter=",")  # Assumes all numeric
speed = np.loadtxt('output/speed2.csv', delimiter=",")  # Assumes all numeric

small = downsample_2d(speed, factor=10 * 10)#*
print(speed.shape)
print(small.shape)
np.savetxt("output/speed_small.csv",small, delimiter=",", fmt='%.6f')
np.savetxt("output/intensity_small.csv", downsample_2d(rank, factor=10 * 10),  delimiter=",",fmt='%.6f')#*


areas = ['oban']
feature_list = ['ndvi','biomass','cover','elevation', 'slope']
resolution = np.float64(0.0001 * 10 * 10) 
x = np.array([8.2,8.7], dtype=np.float64)
y = np.array([5.2,5.7], dtype=np.float64)


df = create_feature_speed_intesity_tables(resolution, areas, feature_list, x, y)
df.to_csv('temp_table.csv', index=False) 





























areas = ['oban']
feature_list = ['ndvi','biomass','cover','elevation', 'slope']
resolution = np.float64(0.0001 * 10 * 10) 
x = np.array([8.2,8.7], dtype=np.float64)
y = np.array([5.2,5.7], dtype=np.float64)


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