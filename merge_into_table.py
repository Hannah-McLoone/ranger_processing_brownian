
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
import rasterio
import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd


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



def block_ids(shape, block_size):
    rows, cols = shape
    r = np.arange(rows) // block_size
    c = np.arange(cols) // block_size
    return (r[:, None] * ((cols + block_size - 1) // block_size) + c[None, :])



def get_park_bbox(park):
    tree = ET.parse(f"gps/{park}.gpx")
    root = tree.getroot()

    lats = []
    lons = []

    # Find all track points
    for trkpt in root.findall(".//default:trkpt", {"default": "http://www.topografix.com/GPX/1/1"}):
        lat = float(trkpt.attrib["lat"])
        lon = float(trkpt.attrib["lon"])
        lats.append(lat)
        lons.append(lon)

    x = np.array([min(lons),max(lons)],dtype=np.float64)
    y = np.array([min(lats), max(lats)],dtype=np.float64)
    return (x,y)


def create_park_table(park, feature_list, year, month, scale_in_metres, block_size = 150):


    #assuming there is a dataset that has been created at this scale for this park
    rank = np.loadtxt(f'/maps/hm708/bb_maps/{park}_{year}_{month}_intensity{scale_in_metres}.csv', delimiter=",")  # Assumes all numeric
    speed = np.loadtxt(f'/maps/hm708/bb_maps/{park}_{year}_{month}_speed{scale_in_metres}.csv', delimiter=",")  # Assumes all numeric


    nrows, ncols = rank.shape

    with rasterio.open(f"/maps/hm708/{park}/{feature_list[0]}.tif") as src:
        transform = src.transform
    rows, cols = np.indices((nrows, ncols))
    lons, lats = rasterio.transform.xy(transform, rows, cols)
    lons = np.array(lons).ravel()
    lats = np.array(lats).ravel()


    #__________ Reproject all feautre rasters to the same fixed dimensions_______
    #my size is every so slightly off with size. (1289, 1052) vs (1290, 1057)
    #i believe some pixel alignment thing. i dont believe to be a big issue !!!!

    features = []
    for f in feature_list:
        feature_as_map = open_and_reproject_to_dimensions(f"/maps/hm708/{park}/{f}.tif", "EPSG:4326", park, speed.shape)
        features.append(feature_as_map.ravel())
    
    features.append([park]*len(features[0]))


    #____combine into one big table____

    #flipping to match - my system was other way round to maybe what standard should be
    rank = rank[::-1]
    speed = speed[::-1]


    # Stack inputs
    X = np.stack(features, axis=1)
    df = pd.DataFrame(X, columns=feature_list + ['park'])

    df['lons'] = lons
    df['lats'] = lats
    df["speed"] = speed.ravel()
    df["rank"] = rank.ravel()
    df["block_id"] = block_ids(rank.shape, block_size).ravel()

    df = df[df['rank'] > 100]
    #add in location
    
    
    df.to_csv(f'/maps/hm708/observations/{park}_{year}_{month}_observations.csv', index=False) 

#assert os.path.isfile(f"{park}_intensity{scale_in_metres}.csv"), f"File does not exist:{park}_intensity{scale_in_metres}.csv"
#assert os.path.isfile(f"{park}_speed{scale_in_metres}.csv"), f"File does not exist: {park}_speed{scale_in_metres}.csv"


for y in range(2016,2025):
    for m in range (1,12):#???
        date = f"{y}-{m:02d}-01"
        feature_list = ["cover", "temperature_"+date, "humidity_"+date, "precipitation_"+date, "elevation", "slope", "biomass"]
        create_park_table('oban', feature_list, y, m, 90)