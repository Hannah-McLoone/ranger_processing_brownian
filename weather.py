import ee
import requests
import zipfile
import io
import xml.etree.ElementTree as ET
import rasterio
import matplotlib.pyplot as plt
import numpy as np

ee.Authenticate()
ee.Initialize(project="friction-maps")



def get_static(image_url: str, scale_in_metres: int, crs: str, region: list[list[float]], layer: str | None = None):
    start_date='2020-01-01' 
    end_date='2020-01-30'
    
    asset_info = ee.data.getAsset(image_url)
    asset_type = asset_info.get("type", "")

    if asset_type == "IMAGE":
        img = ee.Image(image_url)
    elif asset_type == "IMAGE_COLLECTION":
        collection = ee.ImageCollection(image_url)
        if start_date and end_date:
            collection = collection.filterDate(start_date, end_date)
        img = collection.mean()
    else:
        raise ValueError(f"Unsupported asset type: {asset_type}")



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


lons, lat = get_park_bbox('oban')
bbox =  ee.Geometry.Rectangle([lons[0], lat[0], lons[1], lat[1]])


data_bytes = get_static(
    image_url="ECMWF/ERA5_LAND/DAILY_AGGR",
    scale_in_metres=10000,
    crs="EPSG:4326",
    region=bbox,
    layer="dewpoint_temperature_2m"
)

#dewpoint_temperature_2m
#dewpoint_temperature_2m
#temperature_2m

# Open the in-memory GeoTIFF
with rasterio.open(io.BytesIO(data_bytes)) as src:
    arr = src.read(1)  # first band
    bounds = src.bounds
    transform = src.transform

# Mask no-data values
arr = np.ma.masked_equal(arr, src.nodata)

# Plot
plt.figure(figsize=(10, 6))
plt.imshow(arr, cmap="coolwarm", extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
plt.colorbar(label="Temperature (K)")  # ERA5 outputs Kelvin
plt.title("ERA5-Land Daily Mean Temperature (2020-06)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()






































def get_static(image_url: str, scale_in_metres: int, crs: str, region: list[list[float]], layer: str | None = None):
    try:
        img = ee.Image(image_url)
        flag_to_break_try = img.bandNames().getInfo() # this should probabally be implemented better
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
