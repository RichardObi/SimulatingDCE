from pathlib import Path

import numpy as np
import pandas as pd
import h5py
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):

    def __init__(self, args, rtn_idx=False):
        self.args = args
        # set fixed seed
        np.random.seed(42)

        patch_file = Path(args['data_root']) / args['patch_file']
        self.patch_file = patch_file
        self.case_df = pd.read_csv(patch_file.parent / (patch_file.stem + '.csv'))

        # sample (train) or sort (val)
        if self.args['training']:
            self.case_df = self.case_df.sample(frac=1, random_state=42)
        else:
            self.case_df = self.case_df.sort_values(by=['identifier'])
        self.case_df = self.case_df.reset_index(drop=True)

        self.total_count = len(self.case_df)
        if not self.total_count:
            raise ValueError('no cases were found: check label naming')
        print(f'len {self.total_count}')

        self.rtn_idx = rtn_idx
    # ----------------------------------------------------------------
    # augmentation
    # ----------------------------------------------------------------
    @staticmethod
    def rescale_linear(data, scaling):
        [in_low, in_high], [out_low, out_high] = scaling

        m = (out_high - out_low) / (in_high - in_low)
        b = out_low - m * in_low
        data = m * data + b
        data = np.clip(data, out_low, out_high)
        return data

    @staticmethod
    def add_gaussian_noise(data, noise_var):
        variance = np.random.uniform(noise_var[0], noise_var[1])
        data = data + np.random.normal(0.0, variance, size=data.shape)

        return data

    @staticmethod
    def gaussian_blur(data, sigma):
        sig = np.random.uniform(sigma[0], sigma[1])
        return gaussian_filter(data, sig)

    @staticmethod
    def add_offset(data, var):
        off = np.random.uniform(var[0], var[1])
        data = data + off

        return data

    @staticmethod
    def resample(data, xy_zoom, z_zoom):
        fac = [xy_zoom, xy_zoom, z_zoom]

        return zoom(data, zoom=fac, order=1)

    def get_aug_func(self):
        ''' returns a list with all the augmentations '''
        aug = []
        if self.args['augment']['flip']:
            # flip on sagittal plane
            if np.random.rand() < 0.5:
                aug.append(
                    lambda data: np.flip(data, axis=0)
                )
            # flip on coronal plane
            if np.random.rand() < 0.5:
                aug.append(
                    lambda data: np.flip(data, axis=1)
                )

        if self.args['augment']['rot']:
            # rotate k*90 degree
            rot_k = np.random.randint(0, 4)
            if rot_k:
                aug.append(
                    lambda data: np.rot90(
                        data, axes=(0, 1), k=rot_k
                    )
                )

        if self.args['augment']['zoom']:
            xy_zoom = np.random.uniform(
                self.args['augment']['zoom'][0],
                self.args['augment']['zoom'][1],
            )
            z_zoom = np.random.uniform(
                1.,
                self.args['augment']['zoom'][1],
            )
            aug.append(
                lambda data: self.resample(
                    data,
                    xy_zoom,
                    z_zoom,
                )
            )

        if self.args['augment']['offset']:
            aug.append(
                lambda data: self.add_offset(
                    data,
                    self.args['augment']['offset']
                )
            )

        if self.args['augment']['blur']:
            if self.args.augment.blur.type.lower() == 'gaussian':
                aug.append(
                    lambda data: self.gaussian_blur(
                        data,
                        self.args['augment']['blur']['sigma']
                    )
                )
            else:
                raise ValueError('blurring not registered')

        if self.args['augment']['noise']:
            if self.args['augment']['noise']['type'].lower() == 'gaussian':
                aug.append(
                    lambda data: self.add_gaussian_noise(
                        data,
                        self.args['augment']['noise']['variance']
                    )
                )
            else:
                raise ValueError('noise not registered')

        return aug

    def augmentation(self, data, aug_func):
        for func in aug_func:
            data = func(data)
        return data
    # ----------------------------------------------------------------
    # get data
    # ----------------------------------------------------------------
    def get_sequence(self, pid, sequence, slize):
        slize = f'slice_{slize}'
        with h5py.File(self.patch_file, 'r') as ff:
            data = ff[f'{pid}'][slize][sequence][:]

        return data

    def get_data(self, idx):
        row = self.case_df.loc[idx]
        pid = row['identifier']
        slz = row['slice']

        data = []
        for seq in self.args['sequences']:
            data += [self.get_sequence(pid, seq, slz)]
        inputs, *targets = data
        targets = np.array(targets)
        if self.args['subtraction']:
            for i in range(len(targets)):
                targets[i] = np.clip((targets[i] - inputs), a_min=0., a_max=None)
        if self.args['set_input_as_min']:
            for i in range(len(targets)):
                targets[i] = (inputs + np.clip((targets[i] - inputs), a_min=0., a_max=None))

        return inputs[..., np.newaxis], targets[..., np.newaxis]
    # ----------------------------------------------------------------
    # torch Dataset stuff
    # ----------------------------------------------------------------
    def __len__(self):
        ''' cases involved '''
        return self.total_count

    def __getitem__(self, idx, rtn_idx=False):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        inputs, targets = self.get_data(idx)

        if self.rtn_idx:
            return inputs, targets, idx
        return inputs, targets
    # ----------------------------------------------------------------
