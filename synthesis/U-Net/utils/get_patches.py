from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import h5py

# generate patches for train/vaild/test
file_paths = Path('./../data/file_paths/test.csv')
case_df = pd.read_csv(file_paths)
case_df = case_df[~case_df['path_post'].isna()]

patch_file = Path(f'./../data/patches/{file_paths.stem}/patches.h5')
if not patch_file.parent.exists():
    patch_file.parent.mkdir(parents=True)

def get_data(path):
    return np.asarray(Image.open(path)) / 255.

disk_df = pd.DataFrame(columns=['identifier', 'slice'])
with h5py.File(patch_file, 'w') as ff:
    for _, row in case_df.iterrows():
        pid = row['identifier']
        print(pid)
        slz = str(row['slice'])

        pre = get_data(row['path_pre'])
        ff.create_dataset(
            f'{pid}/slice_{slz}/pre',
            data=pre
        )

        post = get_data(row['path_post'])
        for seq in range(post.shape[-1]):
            ff.create_dataset(
                f'{pid}/slice_{slz}/post_{seq}',
                data=post[..., seq].squeeze()
            )

        disk_df.loc[len(disk_df)] = [pid, slz]

disk_df.to_csv(patch_file.parent / (patch_file.stem + '.csv'), index=False)
