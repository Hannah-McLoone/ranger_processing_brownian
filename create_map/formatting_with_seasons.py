import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt



def format_gpx(gpx_file):
    tree = ET.parse(gpx_file)
    root = tree.getroot()

    ns = {'default': 'http://www.topografix.com/GPX/1/1'}

    winter_paths = []
    summer_paths = []
    for track_index, trk in enumerate(root.findall('default:trk', ns)):
        for segment_index, seg in enumerate(trk.findall('default:trkseg', ns)):
            winter_points = []
            summer_points = []
            for trkpt in seg.findall('default:trkpt', ns):
                lat = float(trkpt.attrib['lat'])
                lon = float(trkpt.attrib['lon'])
                time_elem = trkpt.find('default:time', ns)
                time = pd.to_datetime(time_elem.text if time_elem is not None else None)

                entry = {'x': lon,'y': lat,'time': time}

                if time.month in [3,4,5,6,7,8]:
                    summer_points.append(entry)

                else:
                    winter_points.append(entry)


            #all points in a dataframe are of the same track and segment

            if summer_points != []:
                summer_paths.append(pd.DataFrame(summer_points))

            if winter_points != []:
                winter_paths.append(pd.DataFrame(winter_points))
                
    return summer_paths, winter_paths





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
    tree = ET.parse(f"{park}.gpx")
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
