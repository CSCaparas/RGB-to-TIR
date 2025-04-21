import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# file path for folders
tiff_folder = r'C:\Users\Const\OneDrive\Desktop\PhD\Comp_Photo\predictionsPIX2PIX_CC_final'  # folder containing generated .tiff files
csv_folder = r'C:\Users\Const\OneDrive\Desktop\PhD\Comp_Photo\Renamed_Files_paired'  # folder containing round-truth .csv files
save_folder = r"C:\Users\Const\OneDrive\Desktop\PhD\Comp_Photo\groundtruth_final" # folder where ground-truth thermal maps will be saved as .png files

# create list of tiff file names and corresponding stems
tiff_files = [f for f in os.listdir(tiff_folder) if f.endswith('.tiff')]
file_stems = [os.path.splitext(f)[0].replace('_fake', '') for f in tiff_files]

# loop through each stem and generate ground truth image
for stem in file_stems:
    csv_file = stem + '.csv'
    csv_path = os.path.join(csv_folder, csv_file)
    
    if not os.path.exists(csv_path):
        print(f"[WARNING] CSV not found for: {stem}")
        continue
    
    try:
        # read and process CSV (resize to 240 x 240)
        df = pd.read_csv(csv_path, header=None, delimiter=' ', nrows=240*101)
        df = df.iloc[:, 20:260]
        thermal_map = df.values[:240, :]

        plt.imshow(thermal_map, cmap='gray', interpolation='nearest')
        plt.title(f'Thermal Map from {stem}')
        plt.colorbar()

        # save images (as .png files)
        save_path = os.path.join(save_folder, f'{stem}_ground_truth.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {save_path}")

    except Exception as e:
        print(f"[ERROR] Failed for {stem}: {e}")