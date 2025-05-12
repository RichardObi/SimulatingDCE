from pathlib import Path
import pandas as pd

# set path to train/valid/test folder
root_dir = Path('/home/iml/lang/Projects/COBRA/GAN/data/images/test/')

post_df = pd.DataFrame(columns=['identifier', 'slice', 'path'])
for case in (root_dir/f'{root_dir.stem}_B').iterdir():
    *pid, _, slz = case.stem.split('_')
    pid = '_'.join(pid)
    post_df.loc[len(post_df)] = [pid, slz[len('slice'):], str(case)]

pre_df = pd.DataFrame(columns=['identifier', 'slice', 'path'])
for case in (root_dir/f'{root_dir.stem}_A').iterdir():
    *pid, _, slz = case.stem.split('_')
    pid = '_'.join(pid)
    pre_df.loc[len(pre_df)] = [pid, slz[len('slice'):], str(case)]

case_df = pre_df.merge(post_df, on=['identifier', 'slice'], how='left', suffixes=['_pre', '_post'])
case_df.to_csv(f'./../data/file_paths/{root_dir.stem}.csv', index=False)
