import os
import shutil
import time


image_folders = None
#image_folders = ["/home/roo/Desktop/jmi2/test_subtraction/phase_1/prediction", "/home/roo/Desktop/jmi2/test_subtraction/phase_1/targets", "/home/roo/Desktop/jmi2/test_subtraction/phase_2/prediction", "/home/roo/Desktop/jmi2/test_subtraction/phase_2/targets", "/home/roo/Desktop/jmi2/test_subtraction/phase_3/prediction", "/home/roo/Desktop/jmi2/test_subtraction/phase_3/targets","/home/roo/Desktop/jmi2/ldm_data_all_phases/phase1", "/home/roo/Desktop/jmi2/ldm_data_all_phases/phase2", "/home/roo/Desktop/jmi2/ldm_data_all_phases/phase3", "/home/roo/Desktop/jmi2/pre2concat_512p_train/phase1", "/home/roo/Desktop/jmi2/pre2concat_512p_train/phase2", "/home/roo/Desktop/jmi2/pre2concat_512p_train/phase3", "/home/roo/Desktop/jmi2/test_subtraction/inputs", "/home/roo/Desktop/jmi2/gan_output_subtracted/phase1" , "/home/roo/Desktop/jmi2/gan_output_subtracted/phase2", "/home/roo/Desktop/jmi2/gan_output_subtracted/phase3"]

image_folders = ["/home/roo/Desktop/jmi2/gan_output_subtracted/phase1" , "/home/roo/Desktop/jmi2/gan_output_subtracted/phase2", "/home/roo/Desktop/jmi2/gan_output_subtracted/phase3"]


for image_folder in image_folders:
    # === Configuration ===
    mask_folder = "/home/roo/Desktop/jmi2/masks/all"
    unmatched_mask_folder = "/home/roo/Desktop/jmi2/masks/all_unmatched_masks2"


    # === Ensure destination folder exists ===
    os.makedirs(unmatched_mask_folder, exist_ok=True)
    os.makedirs(os.path.join(image_folder, "for_frd"), exist_ok=True)

    # === Gather image keys based on subject ID and slice number ===
    image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg")]
    print(f"Found {len(image_files)} image files.")

    image_keys = set()
    image_files_dict = {}

    for img_file in image_files:
        parts = img_file.replace(".jpg", "").replace(".png", "").replace(".jpeg", "").split("_")
        if len(parts) < 3 or not parts[-1].startswith("slice"):
            print(f"Skipping unexpected image filename format: {img_file}")
            continue
        subject_id = parts[2].zfill(3)
        slice_num = parts[-1].replace("slice", "").zfill(3)
        key = f"{subject_id}_{slice_num}"
        image_keys.add(key)
        #image_files_dict.update({key:img_file})
        image_files_dict[key] = img_file

    print(f"Extracted {len(image_keys)} image keys for matching.")
    #print(f"Extracted {len(image_keys)} image keys for matching. Examples: {image_keys}")

    # === Process mask files ===
    mask_files = [f for f in os.listdir(mask_folder) if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg")]
    unmatched_count = 0
    moved_images_count = 0

    for mask_file in mask_files:
        parts = mask_file.replace(".png", "").split("_")
        if len(parts) < 3 or not parts[-1].startswith("mask"):
            print(f"Skipping unexpected mask filename format: {mask_file}")
            continue
        subject_id = parts[2].zfill(3)
        slice_num = parts[-1].replace("mask", "").zfill(3)
        key = f"{subject_id}_{slice_num}"

        if key not in image_keys:
            # Move unmatched mask
            #src_path = os.path.join(mask_folder, mask_file)
            #dst_path = os.path.join(unmatched_mask_folder, mask_file)
            #shutil.move(src_path, dst_path)
            unmatched_count += 1
            ##print(f"Moved unmatched mask: {mask_file}")
        else:
            # copy image file to new folder
            src_path = os.path.join(image_folder, image_files_dict[key])
            dst_path = os.path.join(image_folder, "for_frd", image_files_dict[key])
            shutil.copy(src_path, dst_path)
            moved_images_count += 1

            # remove key to avoid matching duplicates
            image_keys.remove(key)

    print(f"\nDone. Did not(!!!) move {unmatched_count} unmatched mask files from {mask_folder} to: {unmatched_mask_folder}. Remaining in {mask_folder}: {len(os.listdir(mask_folder))}")
    print(f"Moved {moved_images_count} image files to: {os.path.join(image_folder, 'for_frd')}")
    time.sleep(10)