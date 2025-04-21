import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as compare_ssim

# file paths to folders
generated_folder = r'C:\Users\Const\OneDrive\Desktop\PhD\Comp_Photo\predictionsPIX2PIX_CC_final'
csv_folder = r"C:\Users\Const\OneDrive\Desktop\PhD\Comp_Photo\Renamed_Files_paired"

# output results
results = []

# mean squared error fxn
def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

# loop through all generated .tiff files
for file in os.listdir(generated_folder):
    if file.endswith('.tiff') and '_fake' in file:
        stem = os.path.splitext(file)[0].replace('_fake', '')
        tiff_path = os.path.join(generated_folder, file)
        csv_path = os.path.join(csv_folder, f'{stem}.csv')

        # skip if CSV doesn't exist
        if not os.path.exists(csv_path):
            print(f"[WARNING] CSV not found for: {file}")
            continue

        # load generated image
        gen_img = cv2.imread(tiff_path, cv2.IMREAD_GRAYSCALE)
        if gen_img is None:
            print(f"[ERROR] Could not read generated image: {file}")
            continue

        # load and convert CSV to ground truth image
        try:
            df = pd.read_csv(csv_path, header=None, delimiter=' ', nrows=240 * 101)
            df = df.iloc[:, 20:260]
            thermal_map = df.values[:240, :]
            gt_img = ((thermal_map - thermal_map.min()) / (thermal_map.max() - thermal_map.min()) * 255).astype(np.uint8)
        except Exception as e:
            print(f"[ERROR] Failed to process CSV for {stem}: {e}")
            continue

        # resize ground truth if needed (this shouldn't be required and may cause errors)
        # if gen_img.shape != gt_img.shape:
        #    gt_img = cv2.resize(gt_img, (gen_img.shape[1], gen_img.shape[0]))

        # compute mse and ssim
        mse_score = mse(gen_img, gt_img)
        ssim_score = compare_ssim(gen_img, gt_img)

        results.append({
            'Filename': file,
            'MSE': mse_score,
            'SSIM': ssim_score
        })

        print(f"{file} -> MSE: {mse_score:.4f}, SSIM: {ssim_score:.4f}")

# save results to CSV
df_results = pd.DataFrame(results)
df_results.to_csv('comparison_results.csv', index=False)
print("✅ Comparison results saved to comparison_results.csv")

# suimmarize stats for entire validation set
mean_mse = df_results['MSE'].mean()
mean_ssim = df_results['SSIM'].mean()
median_mse = df_results['MSE'].median()
median_ssim = df_results['SSIM'].median()

print(f"\n📊 Dataset Average — MSE: {mean_mse:.4f}, SSIM: {mean_ssim:.4f}")
print(f"📊 Dataset Median  — MSE: {median_mse:.4f}, SSIM: {median_ssim:.4f}")

# save full report to Excel with summary sheet
summary = pd.DataFrame({
    'Metric': ['Mean', 'Median'],
    'MSE': [mean_mse, median_mse],
    'SSIM': [mean_ssim, median_ssim]
})

with pd.ExcelWriter('comparison_results_summary.xlsx') as writer:
    df_results.to_excel(writer, sheet_name='Per_Image_Results', index=False)
    summary.to_excel(writer, sheet_name='Summary', index=False)