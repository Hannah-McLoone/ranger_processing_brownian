import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ee
import requests
import io
import zipfile
import xml.etree.ElementTree as ET
import os

#currently too big. needs to be a quarter of the size
#check whether puxel alignment it issue
#running this code assumes the tracklog data has already been prcessed at this park and resolution

"""
TO DO:
this is currently only for combined. north southe east and west are ignored
turn esa landcover into a categorical value
change date for ndvi
"""


def get_static(image_url: str, scale_in_metres: int, crs: str, region: list[list[float]], layer: str | None = None):
    try:
        img = ee.Image(image_url)
        flag_to_break_try = img.bandNames().getInfo()
    except:
        img = ee.ImageCollection(image_url).first()

    if layer == 'slope':
        img = img.select('elevation')
        img = ee.Terrain.slope(img)

    elif layer:
        img = img.select(layer)

    url = img.getDownloadURL({
        "scale": scale_in_metres,
        "crs": crs,
        "region": region
    })

    response = requests.get(url)
    if response.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(response.content)) as file:
            with file.open(file.infolist()[0]) as f:
                return f.read()

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






def get_ndvi(region,scale_in_metres, start='2020-01-01', end='2020-01-30'):
    def mask_clouds(image):
        qa = image.select('QA60')
        mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
        return image.updateMask(mask).divide(10000)

    # Build the mean composite
    composite = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                 .filterDate(start, end)
                 .filterBounds(region)
                 .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                 .map(mask_clouds)
                 .mean())

    # Compute NDVI: (B8 - B4)/(B8 + B4)
    ndvi = composite.normalizedDifference(['B8', 'B4']).rename('NDVI')
    url = ndvi.getDownloadURL({
        "scale": scale_in_metres,
        "crs": CRS,
        "region": region
    })
    response = requests.get(url)
    if response.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(response.content)) as file:
            with file.open(file.infolist()[0]) as f:
                return f.read()

    return ndvi





def get_features(park, scale_in_metres):  

    ee.Authenticate()
    ee.Initialize(project="friction-maps")
    CRS = "EPSG:4326"

    lons, lat = get_park_bbox(park)
    bbox =  ee.Geometry.Rectangle([lons[0], lat[0], lons[1], lat[1]]) #[minLon, minLat, maxLon, maxLat]

    for name, (image_url, scale_in_metres, layer) in static.items(): #tqdm(static.items(), desc="static"):
        filename = f"data/{park}/{name}.tif"

        tif = get_static(image_url=image_url, scale_in_metres = scale_in_metres, crs=CRS, region=bbox, layer=layer)
        if tif:
            with open(filename, "wb") as f:
                f.write(tif)

    s2_tif = get_ndvi(bbox, scale_in_metres)
    if s2_tif:
        with open(f"data/{park}/ndvi.tif", "wb") as f:
            f.write(s2_tif)



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




def create_park_table(park, feature_list, scale_in_metres, block_size = 150):
    get_features(park,scale_in_metres) # get the feature table


    #assuming there is a dataset that has been created at this scale for this park
    rank = np.loadtxt(f'{park}_intensity{scale_in_metres}.csv', delimiter=",")  # Assumes all numeric
    speed = np.loadtxt(f'{park}_speed{scale_in_metres}.csv', delimiter=",")  # Assumes all numeric


    #_______________ Reproject all feautre rasters to the same fixed dimensions____________
    #my size is every so slightly off with size. (1289, 1052) vs (1290, 1057)
    #i believe some pixel alignment thing. i dont believe to be a big issue !!!!

    features = []
    for f in feature_list:
        feature_as_map = open_and_reproject_to_dimensions(f"data/{park}/{f}.tif", "EPSG:4326", park, speed.shape)
        features.append(feature_as_map.ravel())
    
    features.append([park]*len(features[0]))


    #____combine into one big table____

    #flipping to match - my system was other way round to maybe what standard should be
    rank = rank[::-1]
    speed = speed[::-1]


    # Stack inputs
    X = np.stack(features, axis=1)
    df = pd.DataFrame(X, columns=feature_list + ['park'])

    df["speed"] = speed.ravel()
    df["rank"] = rank.ravel()
    df["block_id"] = block_ids(rank.shape, block_size).ravel()
    return df




if __name__ == "__main__":
    scale_in_metres = 90 # ---------
    park = 'oban' # --------- # mbe # okwangwo

    assert os.path.isfile(f"{park}_intensity{scale_in_metres}.csv"), f"File does not exist:{park}_intensity{scale_in_metres}.csv"
    assert os.path.isfile(f"{park}_speed{scale_in_metres}.csv"), f"File does not exist: {park}_speed{scale_in_metres}.csv"

    CRS = "EPSG:4326"
    static = {"cover": ("ESA/WorldCover/v200", scale_in_metres, 'Map'),
            "elevation": ("CGIAR/SRTM90_V4", scale_in_metres, 'elevation'),
            "slope": ("CGIAR/SRTM90_V4", scale_in_metres, 'slope'), # slope derived from this
            "biomass": ("NASA/ORNL/biomass_carbon_density/v1", scale_in_metres, 'agb')}
    #ndvi is also included. but it comes from multiple sources so is stated speerately
    feature_list = ['ndvi','biomass','cover','elevation', 'slope']


    df = create_park_table(park, feature_list, scale_in_metres)
    df.to_csv('features_and_output.csv', index=False) 
    
    #in next programs, process rank + get rid of nans, split into train and test using the id
