
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ee
import requests
import io
import zipfile
import xml.etree.ElementTree as ET
import os

#running this code assumes the tracklog data has already been prcessed at this park and resolution

"""
TO DO:
date not hard-coded
this is currently only for combined. north southe east and west are ignored
turn esa landcover into a categorical value
change date for ndvi
"""



def download(img, crs, region,scale_in_metres):
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




def terrain(image_url: str, scale_in_metres: int, crs: str, region: list[list[float]], layer: str | None = None):
    img = ee.Image(image_url)
    img = img.select('elevation')
    if layer == 'slope':
        img = ee.Terrain.slope(img)
    return download(img, crs, region,scale_in_metres
                    )

def others(image_url: str, scale_in_metres: int, crs: str, region: list[list[float]], layer: str | None = None):
    collection = ee.ImageCollection(image_url)
    img = collection.first()
    img = img.select(layer)
    return download(img, crs, region,scale_in_metres)


def weather(start, end, image_url: str, scale_in_metres: int, crs: str, region: list[list[float]], layer: str | None = None,):
    collection = ee.ImageCollection(image_url)
    date_filtered_collection = collection.filterDate(start, end)
    img = date_filtered_collection.mean()
    img = img.select(layer)
    return download(img, crs, region,scale_in_metres)


def get_ndvi(start,end,region, scale_in_metres, crs = "EPSG:4326"):
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
    return download(ndvi, crs, region,scale_in_metres)





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

            




if __name__ == "__main__":
    scale_in_metres = 90 # ---------
    park = 'okwangwo' # ---------  # okwangwo
    ee.Authenticate()
    ee.Initialize(project="friction-maps")
    CRS = "EPSG:4326"

    lons, lat = get_park_bbox(park)
    bbox =  ee.Geometry.Rectangle([lons[0], lat[0], lons[1], lat[1]]) #[minLon, minLat, maxLon, maxLat]

    data_sources= {"cover": ("ESA/WorldCover/v200", scale_in_metres, 'Map'),
            "temperature": ("ECMWF/ERA5_LAND/DAILY_AGGR", scale_in_metres, 'temperature_2m'),
            "humidity": ("ECMWF/ERA5_LAND/DAILY_AGGR", scale_in_metres, 'dewpoint_temperature_2m'),
            "precipitation": ("ECMWF/ERA5_LAND/DAILY_AGGR", scale_in_metres, 'total_precipitation_sum'),
            "elevation": ("CGIAR/SRTM90_V4", scale_in_metres, 'elevation'),
            "slope": ("CGIAR/SRTM90_V4", scale_in_metres, 'slope'), # slope derived from this
            "biomass": ("NASA/ORNL/biomass_carbon_density/v1", scale_in_metres, 'agb')}

    for name, (image_url, scale_in_metres, layer) in data_sources.items():
        if layer == 'Map' or layer == 'agb':
            tif =  others(image_url, scale_in_metres, CRS, bbox, layer)
            open(f"data/{park}/{name}.tif", "wb").write(tif)
            
        elif layer in ['temperature_2m', 'dewpoint_temperature_2m', 'total_precipitation_sum']:

            for year in range(2025, 2026):#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                for month in range(8, 9):
                    start = f"{year}-{month:02d}-01"

                    if month != 12:
                        end = f"{year}-{month+1:02d}-01"
                    else:
                        end = f"{year+1}-01-01"

                    tif =  weather(start, end, image_url, scale_in_metres, CRS, bbox, layer)
                    open(f"data/{park}/{name}_{start}.tif", "wb").write(tif)


        else:
            tif = terrain(image_url, scale_in_metres, CRS, bbox, layer)
            open(f"data/{park}/{name}.tif", "wb").write(tif)



    #ndvi - change this!!!!!!!!!!!!!
    start ='2020-01-01' 
    end='2020-02-01'

    s2_tif = get_ndvi(start, end, bbox, scale_in_metres)
    with open(f"data/{park}/ndvi.tif", "wb") as f:
        f.write(s2_tif)
    
    #in next programs, process rank + get rid of nans, split into train and test using the id