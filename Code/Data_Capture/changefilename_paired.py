import os
import zipfile
import shutil

# === CONFIGURATION ===
zip_file_path = r"C:\Users\Const\OneDrive\Desktop\PhD\3_25_2025.zip"  # Your ZIP archive
extract_dir   = "extracted_files_paired"   # Temp folder for ZIP contents
dest_dir      = "Renamed_Files_paired"     # Where renamed files will go

# === PREPARATION ===
os.makedirs(extract_dir, exist_ok=True)
os.makedirs(dest_dir, exist_ok=True)

# === STEP 1: Extract the archive ===
print(f"Extracting '{zip_file_path}' → '{extract_dir}' …")
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
print("Extraction complete.\n")

# === STEP 2: Walk & group by (folder, base‑name) ===
pairs = {}  # key = (folder_path, base_name), value = dict of {'.csv': path, '.png': path}

for root, dirs, files in os.walk(extract_dir):
    for fname in files:
        ext = os.path.splitext(fname)[1].lower()
        if ext not in ('.csv', '.png'):
            continue
        base = os.path.splitext(fname)[0]
        key = (root, base)
        pairs.setdefault(key, {})[ext] = os.path.join(root, fname)

print(f"Found {len(pairs)} distinct base names (in their subfolders).\n")

# === STEP 3: Move & rename together ===
counter = 1
for (root, base) in sorted(pairs):
    file_dict = pairs[(root, base)]
    # Rename CSV then PNG (if they exist) under the same counter
    for ext in ('.csv', '.png'):
        if ext in file_dict:
            src_path = file_dict[ext]
            new_fname = f"{counter}{ext}"
            dst_path = os.path.join(dest_dir, new_fname)
            print(f"Moving: {src_path!r} → {dst_path!r}")
            try:
                shutil.move(src_path, dst_path)
            except Exception as e:
                print(f"  ⚠️ Error moving {src_path!r}: {e}")
    counter += 1

print("\n✅ All CSV/PNG pairs (and any singletons) have been renamed with the same number and moved.")
