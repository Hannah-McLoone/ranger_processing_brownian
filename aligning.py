import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
import numpy as np
import pandas as pd
#this is currently only for combined. north southe east and west are ignored
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata


"""
turn esa landcover into a categorical value
"""



def find_avg_bounds(files):
    starts = []
    ends = []

    for f in files:
        with rasterio.open(f) as src:
            b = src.bounds
            starts.append((b.left, b.bottom))
            ends.append((b.right, b.top))

    x_bounds = (
        sum(x for x, _ in starts) / len(starts),
        sum(x for x, _ in ends) / len(ends)
    )

    y_bounds = (
        sum(y for _, y in starts) / len(starts),
        sum(y for _, y in ends) / len(ends)
    )
    return x_bounds, y_bounds


    


def open_and_reproject_to_dimensions(file, dst_crs, dst_transform, dst_shape):
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




def create_feature_speed_intesity_tables(resolution, areas, feature_list,x,y):
    for area in areas:

        #x,y = find_avg_bounds([f'data/{area}/{f}.tif' for f in feature_list])
        #x = np.array([x[0],x[1]])
        #y = np.array([y[0],y[1]])

        features = []

        #_______________ Reproject all feautre rasters to the same fixed dimensions____________
        width = int(round(x[1] - x[0],6) / resolution) + 1 # this is the equation used in the stamping code
        height =  int(round(y[1] - y[0],6) / resolution)+ 1
        bounds = (x[0], y[0], x[1], y[1])  # xmin, ymin, xmax, ymax
        dst_transform = from_bounds(*bounds, width=width, height=height)
        dst_shape = (height, width)
        for f in feature_list:
            feature_as_map = open_and_reproject_to_dimensions(f"data/oban/{f}.tif", "EPSG:4326", dst_transform, dst_shape)
            features.append(feature_as_map.ravel())

        #__________________combine into one big table_____________________

        #gpx_file = f'data/{area}/gps.gpx'
        #data = generate_friction_map(x,y,resolution,true_sigma, gpx_file)
        #rank = np.loadtxt('output/intensity_small.csv', delimiter=",")  # Assumes all numeric
        #speed = np.loadtxt('output/speed_small.csv', delimiter=",")  # Assumes all numeric
        speed = np.loadtxt('oban_apeed10.csv', delimiter=",")  # Assumes all numeric

        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


        shape = rank.shape
        size = shape[0] * shape[1]
        rank[rank<20] = 0
        speed = np.where(rank != 0, speed, np.nan)
        #rank = rankdata(rank, method='max').reshape(shape) / size

        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #flipping to match
        #rank = rank[::-1]
        speed = speed[::-1]

        # Stack inputs
        X = np.stack(features, axis=1)
        y = speed.ravel()
        #y2 = rank.ravel()

        # Mask to filter out any rows with NaNs in X or y
        #valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)

        # Apply mask
        X_clean = X#[valid_mask]
        y_clean = y#[valid_mask]
        #y2_clean = y2#[valid_mask]

        df = pd.DataFrame(X_clean, columns=feature_list)
        df["speed"] = y_clean 
        #df["rank"] = y2_clean

        #we now have a datframe of speed, rank and features
        return df

if True:
    areas = ['oban']
    feature_list = ['ndvi','biomass','cover','elevation', 'slope']
    resolution = np.float64(0.0001) 
    x = np.array([8.2,8.7], dtype=np.float64)
    y = np.array([5.2,5.7], dtype=np.float64)


    df = create_feature_speed_intesity_tables(resolution, areas, feature_list, x, y)
    df.to_csv('temp_table.csv', index=False) 