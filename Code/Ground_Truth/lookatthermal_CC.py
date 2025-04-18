import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# file path, filename, and stem
folder    = r'C:\Users\Const\OneDrive\Desktop\PhD\Comp_Photo\Renamed_Files_paired'
file_name = '260.csv'
file_path = os.path.join(folder, file_name)
stem = os.path.splitext(file_name)[0] 

# read in csv (whitespace separated)
df = pd.read_csv(file_path, header=None, delimiter=' ', nrows=240*101)

# resize dataframe to 240 x 240
df = df.iloc[:, 20:260]

# create thermal map
thermal_map = df.values[:240, :]

# plot thermal image (greyscale)
plt.imshow(thermal_map, cmap='gray', interpolation='nearest')
plt.title(f'Thermal Map from {stem}')
plt.colorbar()
plt.show()
