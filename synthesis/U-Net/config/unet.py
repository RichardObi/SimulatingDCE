output:
  output_dir: './runs/mco_subtraction/run'

training:
  loss: 'ssim_mae' # same as MCO
  epochs: 35 # same as MCO
  batch_size: 16 # MCO uses 30
  lr: 0.001 # not specified in MCO
  gamma: 0.99 # not scheduler specified in MCO

model:
  model: 'unet'
  n_time_points: 3 # 5 post-constrast sequences in MCO
  input_channels: 1
  size_first_conv: 64 # same as for MCO

data:
  general:
    sequences: ['pre', 'post_0', 'post_1', 'post_2']
    data_root: './../data/'
    subtraction: True # True in MCO
    set_input_as_min: False # False in MCO
  train:
    patch_file: 'patches/train/patches.h5'
    training: True
  valid:
    patch_file: 'patches/validation/patches.h5'
    training: False
  test:
    patch_file: 'patches/test/patches.h5'
    training: False
