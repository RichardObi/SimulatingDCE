
import cv2 
import numpy as np
import argparse
import os
import glob
from tqdm import tqdm

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concats or extracts grayscale from 3-channel images."
    )
    parser.add_argument(
        "--dataset_path_1",
        type=str,
        default="phase1",
        help="Path to images from first dataset - the postcontrast phase 1 images",
    )
    parser.add_argument(
        "--dataset_path_2",
        type=str,
        default="phase2",
        help="Path to images from second dataset -  the postcontrast phase 2 images",
    )
    parser.add_argument(
        "--dataset_path_3",
        type=str,
        default="phase3",
        help="Path to images from third dataset -  the postcontrast phase 3 images",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="concatenated",
        help="Path to where the concatenated images will be stored.",
    )

    parser.add_argument(
        "--are_phases_in_same_folder",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--reverse_process",
        action="store_true",
        default=False,
        help="If set, the process will be vice versa, i.e. the images from the output_folder will be taken, one image extracted per channel, and stored in dataset_paths 1 to 3.",
    )
    args = parser.parse_args()
    return args

def extract_all_cases(folder_path_1, folder_path_2, folder_path_3, output_folder, are_phases_in_same_folder=False,):
    # get all files from postcontrast image folders using glob


    concatenated_files_names = sorted(glob.glob(output_folder + '/*.png')) if ".png" in os.listdir(output_folder)[0] else sorted(glob.glob(f'{output_folder}/*.jpg'))

    print(f"Found {len(concatenated_files_names)} concatenated images in folder {os.path.abspath(output_folder)}. First example: {os.listdir(output_folder)[0] if len(os.listdir(output_folder)) > 0 else None}")

    # iterate over glob list of images and extract the images from the concatenated channels
    for idx, image_path in tqdm(enumerate(concatenated_files_names)):
    # extract the images from the concatenated channels
        concatenated_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image_phase_1 = concatenated_image[:, :, 0]
        image_phase_2 = concatenated_image[:, :, 1]
        image_phase_3 = concatenated_image[:, :, 2]

        # store each extracted image in the corresponding folder
        if are_phases_in_same_folder:
            folder_path_1 = output_folder
            folder_path_2 = output_folder
            folder_path_3 = output_folder
            os.makedirs(output_folder, exist_ok=True)
        else:
            os.makedirs(folder_path_1, exist_ok=True)
            os.makedirs(folder_path_2, exist_ok=True)
            os.makedirs(folder_path_3, exist_ok=True)

        output_path_phase_1 = os.path.join(folder_path_1, os.path.basename(image_path).replace('_slice', '_0001_slice'))
        output_path_phase_2 = os.path.join(folder_path_2, os.path.basename(image_path).replace('_slice', '_0002_slice'))
        output_path_phase_3 = os.path.join(folder_path_3, os.path.basename(image_path).replace('_slice', '_0003_slice'))

        cv2.imwrite(output_path_phase_1, image_phase_1)
        cv2.imwrite(output_path_phase_2, image_phase_2)
        cv2.imwrite(output_path_phase_3, image_phase_3)


def concat_all_cases(folder_path_1, folder_path_2, folder_path_3, output_folder, are_phases_in_same_folder = False, exists_ok = False, resize_if_size_conflict = True, strict = False):

    # get all files from precontrast image folders using glob
    # TODO: We assume that the files are named (and therefore sorted) in the same order in all folders. This is a strong assumption.

    if are_phases_in_same_folder:
        # in this case all files are in the same folder and the naming convention of the files is different allowing to identify the DCE phase.
        file_names_1 = sorted(glob.glob(folder_path_1 +'/*0001*.png'))
        file_names_2 = sorted(glob.glob(folder_path_1 +'/*0002*.png'))
        file_names_3 = sorted(glob.glob(folder_path_1 +'/*0003*.png'))
        # remove images in file_names_2 that do not have a counterpart in file_names_1
        if not strict:
            print("WARNING: Strict mode is not enabled. This means that we will remove images that do not have a counterpart in the other folders.")
            print(f"file_names_3 (target len): {len(file_names_3)}")
            if len(file_names_1) != len(file_names_3):
                print(f"file_names_1 before: {len(file_names_1)}")
                # Is the file name of phase 1 also present in phase 3 files?
                file_names_1 = [file for file in file_names_1 if file.replace("0001", "0003") in file_names_3]
                print(f"file_names_1 after: {len(file_names_1)}")
            if len(file_names_2) != len(file_names_3):
                print(f"file_names_2 before: {len(file_names_2)}")
                # Is the file name of phase 1 also present in phase 3 files?
                file_names_2 = [file for file in file_names_2 if file.replace("0002", "0003") in file_names_3]
                print(f"file_names_2 after: {len(file_names_2)}")
    else:
        file_names_1 = sorted(glob.glob(folder_path_1 +'/*.png'))
        file_names_2 = sorted(glob.glob(folder_path_2 +'/*.png'))
        file_names_3 = sorted(glob.glob(folder_path_3 +'/*.png'))

    # assert that number of files in all folders is the same
    #if strict:
    assert len(file_names_1) == len(file_names_2) == len(file_names_3), f"Number of files in all folders should be the same. Got {len(file_names_1)}, {len(file_names_2)}, {len(file_names_3)}"
    #else:
    #    if not (len(file_names_1) == len(file_names_2) == len(file_names_3)):
    #        print(f"WARNING: Number of files in all folders should be the same. Got {len(file_names_1)}, {len(file_names_2)}, {len(file_names_3)}.")

    # we iterate over the files in each folder, read them and concatenate them into 3-channel images
    for idx, image_path in tqdm(enumerate(file_names_1)):
        # before reading the images we check if the image names correspond
        assert os.path.basename(file_names_1[idx]).replace("0001", "") == os.path.basename(file_names_2[idx]).replace("0002", "") == os.path.basename(file_names_3[idx]).replace("0003", ""), f"Image names do not correspond: {file_names_1[idx]}, {file_names_2[idx]}, {file_names_3[idx]}"

        # we define storage location of the image in the output folder
        output_file_path = os.path.join(output_folder, os.path.basename(image_path).replace('0001', f'CONCAT'))

        # check if the image already exists and if we should overwrite:
        if os.path.exists(output_file_path):
            if exists_ok:
                continue
            else:
                # we overwrite the image
                print(f'WARNING: {output_file_path} already exists. Overwriting.')

        # read the images in folders 1 to 3 as grayscale images
        image_phase_1 = cv2.imread(file_names_1[idx], cv2.IMREAD_GRAYSCALE)
        image_phase_2 = cv2.imread(file_names_2[idx], cv2.IMREAD_GRAYSCALE)
        image_phase_3 = cv2.imread(file_names_3[idx], cv2.IMREAD_GRAYSCALE)

        # we check if images are same size and resize if allowed with a warning, else we return an exception
        if image_phase_1.shape != image_phase_2.shape or image_phase_1.shape != image_phase_3.shape:
            if resize_if_size_conflict:
                # for upscaling when resizing, INTER_CUBIC is better than INTER_LINEAR. Note, in metrics.py and fid.py INTER_LINEAR is used.
                image_phase_2 = cv2.resize(image_phase_2, image_phase_1.shape, interpolation = cv2.INTER_CUBIC)
                image_phase_3 = cv2.resize(image_phase_3, image_phase_1.shape, interpolation = cv2.INTER_CUBIC)
                print(f'WARNING: Resized images from folder 2 (size:{image_phase_2.shape})  and 3 (size:{image_phase_3.shape}) to match the size of image from folder 1: {image_phase_1.shape}')
            else:
                raise Exception(f'ERROR: Image sizes do not match for counterparts of image {idx} ({image_path}): {image_phase_1.shape}, {image_phase_2.shape}, {image_phase_3.shape}')

        # concatenate the images into a 3-channel image
        concatenated_image = np.stack([image_phase_1, image_phase_2, image_phase_3], axis=-1)

        cv2.imwrite(output_file_path, concatenated_image)


if __name__ == "__main__":
    args = parse_args()
    print(f"args for dce_phase_to_channel_conversion: {args}")

    folder_path_1 = args.dataset_path_1
    folder_path_2 = args.dataset_path_2
    folder_path_3 = args.dataset_path_3
    output_folder = args.output_folder
    are_phases_in_same_folder = args.are_phases_in_same_folder
    reverse_process = args.reverse_process

    os.makedirs(output_folder, exist_ok=True)

    if reverse_process:
        extract_all_cases(folder_path_1, folder_path_2, folder_path_3, output_folder, are_phases_in_same_folder)
    else:
        concat_all_cases(folder_path_1, folder_path_2, folder_path_3, output_folder, are_phases_in_same_folder)


