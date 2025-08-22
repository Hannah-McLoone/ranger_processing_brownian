
import io
import pandas as pd
import pytest
import xml.etree.ElementTree as ET

from create_map.formatting_improved import *


#___________________ Testing format_gpx ____________________
#creating one fake gpx string and parsing it

sample_gpx =  """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="test" xmlns="http://www.topografix.com/GPX/1/1">
  <trk>
    <name>Test Track</name>
    <trkseg>
      <trkpt lat="42.0" lon="-71.0">
        <time>2021-01-01T12:00:00Z</time>
      </trkpt>
      <trkpt lat="43.0" lon="-72.0">
        <time>2021-01-01T13:00:00Z</time>
      </trkpt>
    </trkseg>
    <trkseg>
      <trkpt lat="44.0" lon="-73.0">
        <time>2021-01-01T12:00:00Z</time>
      </trkpt>
      <trkpt lat="45.0" lon="-74.0">
        <time>2021-01-01T13:00:00Z</time>
      </trkpt>
    </trkseg>
  </trk>
</gpx>"""



# Use StringIO so ET.parse can read from it like a file
gpx_file = io.StringIO(sample_gpx)

paths = format_gpx(gpx_file)

# Check we got one dataframe
print(len(paths) == 2)

df = paths[0]
print(df.iloc[0]["x"] == -71.0)
print(df.iloc[0]["y"] == 42.0)
print(pd.Timestamp("2021-01-01T12:00:00Z") == df.iloc[0]["time"])

print(df.iloc[1]["x"] == -72.0)
print(df.iloc[1]["y"] == 43.0)
print(pd.Timestamp("2021-01-01T13:00:00Z") == df.iloc[1]["time"])


df = paths[1]
print(df.iloc[0]["x"] == -73.0)
print(df.iloc[0]["y"] == 44.0)
print(pd.Timestamp("2021-01-01T12:00:00Z") == df.iloc[0]["time"])

print(df.iloc[1]["x"] == -74.0)
print(df.iloc[1]["y"] == 45.0)
print(pd.Timestamp("2021-01-01T13:00:00Z") == df.iloc[1]["time"])







#___________________ Testing haversine_deg_to_met ____________________

#just selecting points and comparing to the results from a third party.
# I used values from this site
#https://www.movable-type.co.uk/scripts/latlong.html


d1 = haversine_deg_to_met(2, 9, 3, 9) / 1000
d2 = haversine_deg_to_met(3, 9, 2, 9) / 1000
print(d1 == d2 and round(d1, 1) == 109.8)


d1 = haversine_deg_to_met(0, 0, 0.01, 0.05) / 1000
d2 = haversine_deg_to_met(0.01, 0.05, 0, 0) / 1000
print(d1 == d2 and round(d1, 2) == 	5.67)


d1 = haversine_deg_to_met(150, 79, -83, -59) / 1000
d2 = haversine_deg_to_met(-83, -59, 150, 79,) / 1000
print(d1 == d2 and round(d1) == 17150)



d1 = haversine_deg_to_met(5.25, 8.25, 5.65, 8.65) / 1000
d2 = haversine_deg_to_met(5.65, 8.65, 5.25, 8.25) / 1000
print(d1 == d2 and round(d1,2) == 62.56 )



d1 = haversine_deg_to_met(8.25, 5.25, 8.65, 5.65) / 1000
d2 = haversine_deg_to_met(8.65, 5.65, 8.25, 5.25) / 1000
print(d1 == d2 and round(d1, 2) == 62.76)


d = haversine_deg_to_met(7.2874, 4.5039, 7.2874, 4.5039) / 1000
print(d == 0)




#_________________Testing directional_distance ___________________


# for any 2 points, (that are not the same) the result should have 2 positive weights, and 2 '0' values
#result should also have weightings whose square sums to 1

value = directional_distance({'prev_x':0, 'prev_y':0, 'x':1, 'y':1})
print(round(sum([i ** 2 for i in value[1:5]]),5) == 1)
print(sum(1 for x in value[1:5] if x > 0) == 2 )#and value.count(0) == 2)

value = directional_distance({'prev_x':1, 'prev_y':1, 'x':1, 'y':1})
[0,0,0,0,0]


value = directional_distance({'prev_x':0, 'prev_y':0, 'x':1, 'y':1})
#print([sqrt(1/2),0, 1/2 ** 0.5, 0])
#print(value[1:5])
#_________________Testing convert_to_speeds_______________________
