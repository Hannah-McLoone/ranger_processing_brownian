import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
zeros = []
b_10 = []
b_20 = []


def downsample_2d(arr, factor=10):
    # Get new shape
    h, w = arr.shape
    new_h, new_w = h // factor, w // factor
    
    # Trim array so it's divisible by factor
    arr = arr[:new_h*factor, :new_w*factor]
    
    # Reshape into blocks
    reshaped = arr.reshape(new_h, factor, new_w, factor)
    
    # Take mean over the 10x10 blocks (ignoring NaNs)
    downsampled = np.nanmean(np.nanmean(reshaped, axis=3), axis=1)
    
    return downsampled

intensity = np.loadtxt(f"oban_intensity10.csv", delimiter=",")
for res in [1,2,3,4,5,6,7,8,9,10,15, 25]:
    values = downsample_2d(intensity, res).ravel()
    zeros.append(100 - (values == 0).sum() / len(values) * 100)
    b_10.append(100 - (values <= 10/ res**2).sum() / len(values) * 100)
    b_20.append(100 - (values <= 20/ res**2).sum() / len(values) * 100)

    print(res)



#percentage of board that is training data
plt.plot([10,20,30,40,50,60,70,80,90,100, 150, 250], zeros, color="red", label = "intensity greater than 0")
plt.plot([10,20,30,40,50,60,70,80,90,100, 150, 250], b_10, color="orange",  label = "intensity greater than 10")
plt.plot([10,20,30,40,50,60,70,80,90,100, 150, 250], b_20, color="yellow",  label = "intensity greater than 20")
plt.xlabel("resolution (m)")
plt.ylabel("percentage of values")
plt.title("percentage of map that is labelled at different resolutions")
plt.show()


