import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim

# === CONFIG ===
generated_dir = r'C:\Users\Const\OneDrive\Desktop\PhD\Comp_Photo\predictionsPIX2PIX_CC_final'
csv_dir       = r'C:\Users\Const\OneDrive\Desktop\PhD\Comp_Photo\Renamed_Files_paired'
out_dir       = r'C:\Users\Const\OneDrive\Desktop\PhD\Comp_Photo\comparison_results'
os.makedirs(out_dir, exist_ok=True)

# Container for metrics
records = []

# === PROCESS ALL GENERATED IMAGES ===
for fname in os.listdir(generated_dir):
    if not fname.endswith('_fake.tiff'):
        continue

    stem = fname.replace('_fake.tiff', '')
    gen_path = os.path.join(generated_dir, fname)
    csv_path = os.path.join(csv_dir, f'{stem}.csv')

    # 1) Load generated image
    gen_img = cv2.imread(gen_path, cv2.IMREAD_GRAYSCALE)
    if gen_img is None:
        print(f"⚠️  Could not read generated image: {gen_path}")
        continue

    # 2) Load & convert ground truth CSV → 8‑bit image
    try:
        df = pd.read_csv(csv_path, header=None, delimiter=' ', nrows=240 * 101)
        df = df.iloc[:, 20:260]
        thermal = df.values[:240, :].astype(float)
        gt_img = ((thermal - thermal.min()) / (thermal.max() - thermal.min()) * 255).astype(np.uint8)
    except Exception as e:
        print(f"⚠️  Error reading CSV for {stem}: {e}")
        continue

    # 3) Compute MSE & SSIM
    mse  = np.mean((gen_img.astype(float) - gt_img.astype(float))**2)
    ssim = compare_ssim(gen_img, gt_img)

    # 4) Plot side‑by‑side
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(gt_img,  cmap='gray')
    axs[0].set_title('Ground Truth')
    axs[0].axis('off')

    axs[1].imshow(gen_img, cmap='gray')
    axs[1].set_title('Generated')
    axs[1].axis('off')

    fig.suptitle(f'{stem}  —  MSE: {mse:.4f}  |  SSIM: {ssim:.4f}', fontsize=14)
    plt.tight_layout(rect=[0,0,1,0.95])

    # 5) Save figure
    out_path = os.path.join(out_dir, f'{stem}_comparison.png')
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"✅ Saved comparison for {stem} → {out_png}")
