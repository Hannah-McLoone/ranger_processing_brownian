import ee
import requests
import io
import zipfile
import xml.etree.ElementTree as ET
import numpy as np

"""
to do:
change date for ndvi
change scale
"""


ee.Authenticate()
ee.Initialize(project="friction-maps")
scale = 100
CRS = "EPSG:4326"




static = {"cover": ("ESA/WorldCover/v200", scale, 'Map'),
          "elevation": ("CGIAR/SRTM90_V4", scale, 'elevation'),
          "slope": ("CGIAR/SRTM90_V4", scale, 'slope'), # slope derived from this
          "biomass": ("NASA/ORNL/biomass_carbon_density/v1", scale, 'agb')}
#ndvi


def get_static(image_url: str, scale: int, crs: str, region: list[list[float]], layer: str | None = None):
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
        "scale": scale,
        "crs": crs,
        "region": region
    })

    response = requests.get(url)
    if response.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(response.content)) as file:
            with file.open(file.infolist()[0]) as f:
                print(3)
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






def get_ndvi(region, start='2020-01-01', end='2020-01-30'):
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
        "scale": scale,
        "crs": CRS,
        "region": region
    })
    response = requests.get(url)
    if response.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(response.content)) as file:
            with file.open(file.infolist()[0]) as f:
                return f.read()

    return ndvi














def pipeline(park):  
    lons, lat = get_park_bbox(park)
    bbox =  ee.Geometry.Rectangle([lons[0], lat[0], lons[1], lat[1]]) #[minLon, minLat, maxLon, maxLat]

    for name, (image_url, scale, layer) in static.items(): #tqdm(static.items(), desc="static"):
        filename = f"data/{park}/{name}.tif"

        tif = get_static(image_url=image_url, scale=scale, crs=CRS, region=bbox, layer=layer)
        if tif:
            with open(filename, "wb") as f:
                f.write(tif)

    s2_tif = get_ndvi(bbox)
    if s2_tif:
        with open(f"data/{park}/ndvi.tif", "wb") as f:
            f.write(s2_tif)


#pipeline('oban')