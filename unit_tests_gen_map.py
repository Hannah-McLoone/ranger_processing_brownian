from create_map.generate_map import *
import matplotlib.pyplot as plt
import scipy.stats as stats
import random
import statistics

# standard deviation vs intensity!!!!!!!!!!!
# my normal turned into a chai- squared test



#_________________________Testing brownian bridge ___________________________________


start_coord = (0,0)
start_time = 0
end_coord = (0,0)
end_time = 1
standard_deviation = 1


#TEST 0
# test that it starts and ends in the correct place
values =  brownian_bridge((start_time,start_coord[0] ), (end_time,end_coord[0]),standard_deviation, 100)
print("TEST 0: ", values[0] == start_coord[0] and values[-1] == end_coord[0])

values_x, values_y = brownian_2d(start_coord,end_coord, start_time, end_time, standard_deviation, 100)
print("TEST 0: ", (values_x[0], values_y[0]) == start_coord and (values_x[-1], values_y[-1]) == end_coord)

#TEST 1
# test that the steps are guassian
d_list = []
for n in range (1,50):
    values =  brownian_bridge((start_time,start_coord[0] ), (end_time,end_coord[0]),standard_deviation, 100)
    differences = [values[i+1] - values[i] for i in range(len(values)-1)]
    d_list  = np.append(d_list, differences)

stat, p = stats.shapiro(differences)
print("TEST 1: ",p > 0.05)


#TEST 2
#test that cross-section is gaussian
values = []
cut = 50
for n in range (1,1000):
    values.append(brownian_bridge((start_time,start_coord[0]), (end_time,end_coord[0]),standard_deviation, 100)[cut])
stat, p = stats.shapiro(values)
print("TEST 2: ",p > 0.05)
sd = statistics.stdev(values)
t = cut / 100
print("TEST 4: ", round(sd,1) == round((t * (1-t))**0.5, 1))



#TEST 3
values = []
cut = 20
for n in range (1,1000):
    values.append(brownian_bridge((start_time,start_coord[0]), (end_time,end_coord[0]),standard_deviation, 100)[cut])
stat, p = stats.shapiro(values)
print("TEST 3: ",p > 0.05)

#square root, t, 1-t. * variance
sd = statistics.stdev(values)
t = cut / 100
print("TEST 4: ", round(sd,1) == round((t * (1-t))**0.5, 1))

#TEST 3 - FAIL
#test that bb steps are normally distributed in 2d
#values = brownian_2d(start_coord,end_coord, start_time, end_time, standard_deviation, 1000)
#differences = [((values[0][i+1] - values[0][i]) **2 + (values[1][i+1] - values[1][i]) **2)**0.5 for i in range(len(values[0])-1)]
#stat, p = stats.shapiro(differences)
#print("TEST 3: ",p > 0.05)

#test that cross section is actually normally distributed
#values = brownian_2d(start_coord,end_coord, start_time, end_time, standard_deviation,100)




#______________________ testing the grid_map object___________________


# -------starting off with a very simplified setup:-------
simple_map = grid_map.__new__(grid_map)
simple_map.resolution_x = 1
simple_map.resolution_y = 1
simple_map.x_range = [0, 3]
simple_map.y_range = [0, 3]
simple_map.set_up()
# i have gone with a system that is inclusive on both ends, so this will have a width of 4


# test .increase method - very simple
#these values for NSEW are not possible - just testing values
simple_map.increase((2,2), 1, [0.8,0,0.2,0])
expected = np.array([[0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0]])
print("TEST 4: ",(simple_map.general == expected).all())
print("TEST 5: ",(simple_map.north == 0.8 * expected).all())
print("TEST 6: ",(simple_map.south == 0 * expected).all())
print("TEST 7: ",(simple_map.east == 0.2 * expected).all())
print("TEST 8: ",(simple_map.west == 0 * expected).all())



# increases multiple pixels, multiple times, with different weightings
simple_map.set_up()
simple_map.increase((0,0), 3, [0,0,0,1])
simple_map.increase((3,2), 1, [0,0,0,0.5])
simple_map.increase((3,2), 1, [0,0,0,0.3])
expected = np.array([[3,0,0,0],[0,0,0,0],[0,0,0,2],[0,0,0,0]])
print("TEST 9: ",(simple_map.general == expected).all())

expected_west = np.array([[3,0,0,0],[0,0,0,0],[0,0,0,0.8],[0,0,0,0]])
print("TEST 10: ",(simple_map.west == expected_west).all())





# -------now a more complicated setup:-------
#the resolution is not 1
#and the range does not start at 0


simple_map = grid_map.__new__(grid_map)
simple_map.resolution_x = 0.25
simple_map.resolution_y = 0.2
simple_map.x_range = [5, 10]
simple_map.y_range = [-2, 4]
simple_map.set_up()

expected = np.zeros((31, 21))

simple_map.increase((5,-2), 1, [0,0,0,0])
simple_map.increase((10,0.01), 1, [0,0,0,0])
simple_map.increase((7.250001,0.40001), 1, [0,0,0,0])

expected[0,0] = 1
expected[10,-1] = 1
expected[12,9] = 1
print("TEST 11: ",(simple_map.general == expected).all())











#__________________Testing the program as a whole ___________________________

#testing all of these things combined


"""
DOES NOT WORK 

fake_map = grid_map.__new__(grid_map)

# Manually set only the attributes needed by `increase`
fake_map.resolution_x = 1
fake_map.resolution_y = 1
fake_map.x_range = [0, 100]
fake_map.y_range = [0, 100]
fake_map.width = 100
fake_map.height = 100
fake_map.general = np.zeros((fake_map.height, fake_map.width))

row = {'north':1,
       'south':1,
       'east':1,
       'west':1,
       'time':pd.to_datetime("2025-08-21 10:00:00"),
       'prev_time':pd.to_datetime("2025-08-21 09:59:00"),
       'x':50,
       'prev_x':50,
       'y':10,
       'prev_y':90,
       'speed_kmph':5}

connect_points_with_bb_simulations(row, 3, fake_map, fake_map, 10000)

stat, p = stats.shapiro(fake_map.general[20])
plt.plot(fake_map.general[20])
plt.show()
print(p > 0.05)

#plt.imshow(fake_map.general, cmap='hot')
#plt.colorbar()
#plt.show()
"""