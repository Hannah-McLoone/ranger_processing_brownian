import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
from datetime import datetime
#need to edit so that the years are present are not hard-coded in
# (right now 2016 to 2024)




def format_gpx(gpx_file):
    tree = ET.parse(gpx_file)
    root = tree.getroot()

    ns = {'default': 'http://www.topografix.com/GPX/1/1'}
    years = get_park_years(gpx_file)
    paths = []
    for year_index in range (0,years[1]-years[0]+1):
        paths.append([])
        for month in range (0,12):
            paths[year_index].append([])


    for track_index, trk in enumerate(root.findall('default:trk', ns)):
        for segment_index, seg in enumerate(trk.findall('default:trkseg', ns)):

            points = []
            for year_index in range (0,years[1]-years[0]+1):
                points.append([])
                for month in range (0,12):
                    points[year_index].append([])

            for trkpt in seg.findall('default:trkpt', ns):
                lat = float(trkpt.attrib['lat'])
                lon = float(trkpt.attrib['lon'])
                time_elem = trkpt.find('default:time', ns)
                time = pd.to_datetime(time_elem.text if time_elem is not None else None)
                year_index = time.year -  years[0]

                entry = {'x': lon,'y': lat,'time': time}

                if year_index>=0:#there are a couple random entries from 1970
                    points[year_index][time.month - 1].append(entry)


            #all points in a dataframe are of the same track and segment
            for year_index in range (0,len(paths)):
                for month in range (0,12):
                    if points[year_index][month] != []:
                        paths[year_index][month].append(pd.DataFrame(points[year_index][month]))

                
    return paths





def haversine_deg_to_met(lon1, lat1, lon2, lat2):
    #(x,y) (lon, lat)
    R = 6371000  # Earth radius in meters
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c


def directional_distance(row):
    lat1, lon1, lat2, lon2 = row['prev_y'], row['prev_x'], row['y'], row['x']

    if pd.isnull(lat1):
        return pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan])
    
    # Latitude and longitude differences
    lat_diff = lat2 - lat1
    lon_diff = lon2 - lon1


    # Compute individual directional distances
    #because i like the x,y convention
    #(x,y) (lon, lat)
    distance = haversine_deg_to_met(lon1,lat1, lon2,lat2)

    dist_north = haversine_deg_to_met(lon1,lat1, lon1,lat2) if lat_diff > 0 else 0
    dist_south = haversine_deg_to_met(lon1,lat1, lon1,lat2) if lat_diff < 0 else 0
    dist_east  = haversine_deg_to_met(lon1,lat1, lon2,lat1) if lon_diff > 0 else 0
    dist_west  = haversine_deg_to_met(lon1,lat1, lon2,lat1) if lon_diff < 0 else 0

    # Total distance to normalize
    total = dist_north**2 + dist_south**2 + dist_east**2 + dist_west**2

    # Avoid division by zero
    if total == 0:
        return pd.Series([0,0,0,0,0])

    # Relative distances - unit vector components
    rel_north = dist_north / sqrt(total)
    rel_south = dist_south / sqrt(total)
    rel_east  = dist_east / sqrt(total)
    rel_west  = dist_west / sqrt(total)

    # Format as strings with direction
    return pd.Series([distance, rel_north,rel_south,rel_east,rel_west])


def convert_to_speeds(df):

    # Shift coordinates and time to compute differences
    df['prev_x'] = df['x'].shift()
    df['prev_y'] = df['y'].shift()
    df['prev_time'] = df['time'].shift()

    # Compute distance in meters
    df[['distance_m','north', 'south', 'east', 'west']]  = df.apply(
        lambda row: directional_distance(row),
        axis=1
    )

    # Compute time difference in seconds
    df['time_diff_s'] = (df['time'] - df['prev_time']).dt.total_seconds()

    # Compute speed (meters per second)
    df['speed_mps'] = df['distance_m'] / df['time_diff_s']
    df['speed_kmph'] = df['speed_mps'] * 3.6

    #drop intermediate columns!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    df.drop(columns=['speed_mps'], inplace=True) # distance_m, time_diff_s
    return df # with n s e w columns





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




def get_park_years(gpx_file):
    ns = {"gpx": "http://www.topografix.com/GPX/1/1"}
    root = ET.parse(gpx_file).getroot()

    years = []
    for trkpt in root.findall(".//gpx:trkpt", ns):
        t = trkpt.find("gpx:time", ns)
        if t is not None and t.text:
            try:
                years.append(datetime.fromisoformat(t.text.replace("Z", "+00:00")).year)
            except ValueError:
                pass

    return (max(min(years),2016), max(years)) if years else None



#df_list = format_gpx('gps/oban.gpx')
#print(df_list)