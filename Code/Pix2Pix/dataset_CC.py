#%%
from glob import glob
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pandas as pd

#%%
class RGB2ThermalDataset(Dataset):
    def __init__(self, rgb_pattern, thermal_pattern, transform=None):
        self.rgb_paths     = sorted(glob(rgb_pattern,    recursive=True))
        self.thermal_paths = sorted(glob(thermal_pattern,recursive=True))
        assert len(self.rgb_paths) == len(self.thermal_paths), \
            f"#{len(self.rgb_paths)} RGB vs #{len(self.thermal_paths)} CSVs"
        self.transform = transform

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        # load RGB and get its original size 
        rgb = Image.open(self.rgb_paths[idx]).convert("RGB")
        w, h = rgb.size  # width, height

        # load the thermal map from CSV (whitespace‑separated)
        import numpy as np
        tir_df = pd.read_csv(self.thermal_paths[idx], header=None, delimiter=' ', nrows=240*101)  # no delimiter=','
        
        # changes thermal map from 240 x 320 to 240 x 240 (to match RGB), converts to uint8
        tir = tir_df.iloc[0:240, 20:260].values.astype('uint8')

        # convert to tensors: RGB (3×H×W), TIR (1×H×W) 
        rgb = TF.to_tensor(rgb)
        tir = TF.to_tensor(tir)

        return rgb, tir, idx


    def get_filenames(self, idx):
        return self.thermal_paths[idx]
