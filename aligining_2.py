import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata
from pipeline import pipeline, get_park_bbox

#currently too big. needs to be a quarter of the size

"""
this is currently only for combined. north southe east and west are ignored
turn esa landcover into a categorical value
"""


def open_and_reproject_to_dimensions(file, dst_crs, park, shape):
    # this function reprojects the physical feature to be compatible with the tracklog map
    # the feature is the source 
    # the brownian tracklog map is the destination format

    
    x,y = get_park_bbox(park) #bounds of tracklog data
    dst_bounds = (x[0], y[0], x[1], y[1])  # xmin, ymin, xmax, ymax

    dst_width, dst_height = shape[1], shape[0] # chape of tracklog - check this!!!!!!!!!!!!!!

    dst_transform = from_bounds(*dst_bounds, width=dst_width, height=dst_height)
    dst_shape = (dst_height, dst_width)

    with rasterio.open(file) as src:
        src_array = src.read(1)

        # Ensure float dtype so NaN is possible
        reprojected_array = np.full(dst_shape, np.nan, dtype=np.float32)

        reproject(
            source=src_array.astype(np.float32),  # convert to float
            destination=reprojected_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
            dst_nodata=np.nan   # explicitly say nodata is NaN
        )

        #plt.imshow(reprojected_array, cmap='hot')
        #plt.colorbar()
        #plt.show()

    return reprojected_array







def create_park_table(park, feature_list):

    scale_in_metres = 10
    pipeline(park,scale_in_metres) # get the feature table


    #assuming there is a dataset that has been created at this scale for this park
    #rank = np.loadtxt(f'oban_intensity{scale_in_metres}.csv', delimiter=",")  # Assumes all numeric
    speed = np.loadtxt(f'oban_speed{scale_in_metres}.csv', delimiter=",")  # Assumes all numeric


    #_______________ Reproject all feautre rasters to the same fixed dimensions____________

    
    features = []
    for f in feature_list:
        if f"data/oban/{f}.tif": # check <- has these dimensions and bounds
            feature_as_map = open_and_reproject_to_dimensions(f"data/oban/{f}.tif", "EPSG:4326", park, speed.shape())
            print('bad')
            print(0/0)
        else:
            feature_as_map = f"data/oban/{f}.tif"
            print('good')
            print(0/0)
        features.append(feature_as_map.ravel())

    #__________________combine into one big table_____________________

    #flipping to match
    rank = rank[::-1]
    speed = speed[::-1]

    # Stack inputs
    X = np.stack(features, axis=1)
    df = pd.DataFrame(X, columns=feature_list)
    df["speed"] = speed.ravel()
    df["rank"] = rank.ravel()

    return df





feature_list = ['ndvi','biomass','cover','elevation', 'slope']
df = create_park_table('oban', feature_list)
df.to_csv('temp_table.csv', index=False) 
#process rank + get rid of nans