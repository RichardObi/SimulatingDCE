import os
import cv2
import numpy as np

# === CONFIGURATION ===
phase = "2"
pre_folder = f"/home/roo/Desktop/jmi2/test_subtraction/inputs/"
post_folder = f"/home/roo/Desktop/jmi2/pre2concat_512p_train/phase{phase}/"
output_folder = f"/home/roo/Desktop/jmi2/gan_output_subtracted/phase{phase}/"

# === Ensure output folder exists ===
os.makedirs(output_folder, exist_ok=True)

def extract_id_and_slice(filename):
    """Extract subject ID and slice number from filename"""
    parts = filename.replace(".jpg", "").replace(".jpeg", "").replace(".png", "").split("_")
    if len(parts) < 3 or not parts[-1].startswith("slice"):
        return None
    subject_id = parts[2].zfill(3)
    slice_num = parts[-1].replace("slice", "").zfill(3)
    return f"{subject_id}_{slice_num}"

# === Index precontrast images by key ===
pre_files = [f for f in os.listdir(pre_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
pre_index = {}

print(f"Indexing {len(pre_files)} precontrast images...")

for f in pre_files:
    key = extract_id_and_slice(f)
    if key:
        pre_index[key] = f
        print(f"Indexed precontrast image: {f} → key: {key}")
    else:
        print(f"Skipped invalid precontrast filename: {f}")

# === Process postcontrast images ===
post_files = [f for f in os.listdir(post_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
matched = 0
skipped = 0

print(f"\nProcessing {len(post_files)} postcontrast images...")

for post_file in post_files:
    key = extract_id_and_slice(post_file)
    if not key:
        print(f"[SKIP] Invalid postcontrast filename: {post_file}")
        skipped += 1
        continue

    if key not in pre_index:
        print(f"[SKIP] No matching precontrast image for key: {key}")
        skipped += 1
        continue

    pre_file = pre_index[key]
    print(f"[MATCH] {pre_file} (pre) ↔ {post_file} (post)")

    # Load images (grayscale)
    pre_path = os.path.join(pre_folder, pre_file)
    post_path = os.path.join(post_folder, post_file)

    pre_img = cv2.imread(pre_path, cv2.IMREAD_GRAYSCALE)
    post_img = cv2.imread(post_path, cv2.IMREAD_GRAYSCALE)

    if pre_img is None or post_img is None:
        print(f"[ERROR] Could not load one or both images for key: {key}")
        skipped += 1
        continue

    if pre_img.shape != post_img.shape:
        print(f"[SKIP] Shape mismatch for key: {key} ({pre_img.shape} vs {post_img.shape})")
        skipped += 1
        continue

    # Subtract and save
    diff_img = cv2.subtract(post_img, pre_img)
    out_path = os.path.join(output_folder, post_file)

    cv2.imwrite(out_path, diff_img)
    print(f"[SAVED] Subtracted image written to: {out_path}")
    matched += 1

print(f"\nDone. Matched: {matched}, Skipped: {skipped}")
