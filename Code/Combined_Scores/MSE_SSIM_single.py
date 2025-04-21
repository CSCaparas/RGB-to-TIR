import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim

# set stem (image number)
stem = "3044"

# file paths to images
generated_path = fr'C:\Users\Const\OneDrive\Desktop\PhD\Comp_Photo\predictionsPIX2PIX_CC_final\{stem}_fake.tiff'
csv_path = fr'C:\Users\Const\OneDrive\Desktop\PhD\Comp_Photo\Renamed_Files_paired\{stem}.csv'

# load generated .tiff image
gen_img = cv2.imread(generated_path, cv2.IMREAD_GRAYSCALE)
if gen_img is None:
    raise FileNotFoundError(f"❌ Failed to load generated image: {generated_path}")

# load ground truth CSV and convert to grayscale image
try:
    df = pd.read_csv(csv_path, header=None, delimiter=' ', nrows=240 * 101)
    df = df.iloc[:, 20:260]
    thermal_map = df.values[:240, :]

    # normalize to 8-bit grayscale
    gt_img = ((thermal_map - thermal_map.min()) / (thermal_map.max() - thermal_map.min()) * 255).astype(np.uint8)

except Exception as e:
    raise RuntimeError(f"❌ Error processing CSV {csv_path}: {e}")

# resize ground truth if needed (this shouldn't be required and may cause errors)
# if gen_img.shape != gt_img.shape:
#    gt_img = cv2.resize(gt_img, (gen_img.shape[1], gen_img.shape[0]))

# compute mse and ssim
mse = np.mean((gen_img.astype("float") - gt_img.astype("float")) ** 2)
ssim, _ = compare_ssim(gen_img, gt_img, full=True)

# show side-by-side comparison
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(gt_img, cmap='gray')
axs[0].set_title('Ground Truth')
axs[0].axis('off')

axs[1].imshow(gen_img, cmap='gray')
axs[1].set_title('Generated')
axs[1].axis('off')

fig.suptitle(f'{stem}: MSE: {mse:.4f} | SSIM: {ssim:.4f}', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
