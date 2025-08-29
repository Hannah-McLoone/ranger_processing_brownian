import numpy as np
import pandas as pd
#from scipy.stats import rankdata
from formatting_improved import format_gpx, convert_to_speeds, get_park_bbox
from determining_sigma import determine_sigmas
import sys
import math
import matplotlib.pyplot as plt

#the code for the following class has been taken from this article
#https://medium.com/@christopher.tabori/bridging-the-gap-an-introduction-to-brownian-bridge-simulations-10655b0baf02
#not to be used on the poles - will result in error

class grid_map:
    def __init__(self, resolution_in_metres, x_range, y_range):
        degree_lat_per_metre = 1 / 111320
        degree_lon_per_metre = 1 / (111320.0 * math.cos(math.radians(y_range[0])))

        self.resolution_x = resolution_in_metres * degree_lon_per_metre
        self.resolution_y = resolution_in_metres * degree_lat_per_metre
        self.x_range = x_range
        self.y_range = y_range
        self.set_up()

    def set_up(self):
        self.width = int((self.x_range[1] - self.x_range[0]) / self.resolution_x) + 1
        self.height = int((self.y_range[1] -self. y_range[0] )/ self.resolution_y) + 1 #???????

        self.general = np.zeros((self.height, self.width))
        self.north = np.zeros((self.height, self.width))
        self.south = np.zeros((self.height, self.width))
        self.east = np.zeros((self.height, self.width))
        self.west = np.zeros((self.height, self.width))


    def increase(self, data_point, value, direction_weights):
        n,s,e,w = direction_weights
        x = int((data_point[0] - self.x_range[0])//self.resolution_x)
        y = int((data_point[1] - self.y_range[0])//self.resolution_y)

        if y < self.height and x < self.width and x>=0 and y >=0:
            self.general[y,x] += value

            
            self.north[y,x] += value * n
            self.south[y,x] += value * s
            self.east[y,x] += value * e
            self.west[y,x] += value * w


def brownian_bridge_old(start, end, standard_deviation: float=1.0,num_intermediate_points: int=1024):
    time_steps = np.linspace(0, 1, num=num_intermediate_points+1)
    scale = np.sqrt(1/float(num_intermediate_points))
    deterministic = start + time_steps*(end - start)
    noise = scale*standard_deviation*np.random.normal(size=num_intermediate_points)
    noise = np.insert(noise, [0], 0)
    browinian_motion = np.cumsum(noise)
    return deterministic + browinian_motion - (browinian_motion[-1])*time_steps


    

def brownian_bridge(start, end, standard_deviation,num_intermediate_points):
    time_steps = np.linspace(0, 1, num=num_intermediate_points+1)
    scale = np.sqrt((end[0]-start[0])/float(num_intermediate_points))
    deterministic = start[1] + time_steps*(end[1] - start[1])
    noise = scale*standard_deviation*np.random.normal(size=num_intermediate_points)
    noise = np.insert(noise, [0], 0)
    browinian_motion = np.cumsum(noise)
    return deterministic + browinian_motion - (browinian_motion[-1])*time_steps


def brownian_2d(start_coord,end_coord, start_time, end_time, standard_deviation, num_intermediate_points):#x,y
    data_x = brownian_bridge((start_time, start_coord[0]),(end_time, end_coord[0]), standard_deviation, num_intermediate_points)
    data_y = brownian_bridge((start_time, start_coord[1]),(end_time, end_coord[1]), standard_deviation, num_intermediate_points)
    return data_x, data_y



def connect_points_with_bb_simulations(row, variance, intensity, speed, number_of_runs=100, num_intermediate_points = 100):
    n = row['north']
    s = row['south']
    e = row['east']
    w = row['west']
    prev_time = pd.to_datetime(row['prev_time']).timestamp()
    time = pd.to_datetime(row['time']).timestamp()
    for _ in range (0,number_of_runs):
        data_x, data_y = brownian_2d((row['prev_x'],row['prev_y']), (row['x'],row['y']),  prev_time, time, variance, num_intermediate_points)
        for j in range(0,len(data_x)):
            #speed.increase((data_x[j],data_y[j]),row['speed_kmph'],[n,s,e,w])
            intensity.increase((data_x[j],data_y[j]),1,[n,s,e,w])


        #plt.imshow(intensity.general, cmap='hot')
        #plt.colorbar()
        #plt.show()



def run_main(x_range, y_range, resolution_in_metres, number_of_runs,paths, variance_list):
    speed = grid_map(resolution_in_metres, x_range, y_range)
    intensity = grid_map(resolution_in_metres, x_range, y_range)

    #if speed does not = nan
    for path_counter in range (0,len(paths)):
        for idx, row in paths[path_counter].iterrows():
            if row['speed_kmph'] < 10 and row['speed_kmph']>0 and pd.to_datetime(row['time']) - pd.to_datetime(row['prev_time']) < pd.Timedelta(minutes=15):
                variance = variance_list[path_counter]
                connect_points_with_bb_simulations(row,variance,intensity, speed)


    
    nan_grid = np.full_like(intensity.general, np.nan, dtype=np.float64)
    speed.general = np.divide(speed.general, intensity.general, out=nan_grid, where=intensity.general != 0)
    speed.north = np.divide(speed.north, intensity.north, out=nan_grid, where=intensity.north != 0)
    speed.south = np.divide(speed.south, intensity.south, out=nan_grid, where=intensity.south != 0)
    speed.east = np.divide(speed.east, intensity.east, out=nan_grid, where=intensity.east != 0)
    speed.west = np.divide(speed.west, intensity.west, out=nan_grid, where=intensity.west != 0)

    return speed, intensity











if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <park> <resolution_in_metres>")
        sys.exit(1)
    

    park = sys.argv[1] # mbe, okwangwo
    resolution_in_metres = int(sys.argv[2]) # 10?
    number_of_runs = 100#----------------------------------




    x_range, y_range = get_park_bbox(park)

    gpx_file = f'{park}.gpx' # put in folder?
    #summer_df_list, winter_df_list = format_gpx(gpx_file)
    df_list = format_gpx(gpx_file)


    #do this for winter too:
    paths = []
    for df in df_list:
        paths.append(convert_to_speeds(df))


    variance_list = determine_sigmas(paths)

    speed, intensity = run_main(x_range, y_range, resolution_in_metres, number_of_runs, paths, variance_list)

    np.savetxt(f"oban_speed{resolution_in_metres}.csv", speed.general, delimiter=",", fmt='%.6f')
    np.savetxt(f"oban_intensity{resolution_in_metres}.csv", intensity.general, delimiter=",", fmt='%.6f')

