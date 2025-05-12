import argparse
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
#from torchmetrics.functional import structural_similarity_index_measure

from src import dataset
from src import metrics
from src.utils import get_logdir, copy_files

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# -------------------------------------------------------------------------------
# read config file
# -------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('config_file', help='yaml config file')
pargs = parser.parse_args()

#config_file = './config.yml'
with open (pargs.config_file, 'r') as ff:
    par = ff.read()
    args = yaml.safe_load(par)

log_dir = get_logdir(args['output']['output_dir'])
print(f'logging at {log_dir}')
Path(log_dir).mkdir(parents=True)

# -------------------------------------------------------------------------------
# dataloader
# -------------------------------------------------------------------------------

train_args = {
    **args['data']['general'],
    **args['data']['train'],
    'input_channels': args['model']['input_channels'],
    'n_time_points': args['model']['n_time_points'],
}
train_ds = dataset.CustomDataset(train_args)

val_args = {
    **args['data']['general'],
    **args['data']['valid'],
    'input_channels': args['model']['input_channels'],
    'n_time_points': args['model']['n_time_points'],
}
val_ds = dataset.CustomDataset(val_args)

train_loader = DataLoader(
    train_ds,
    batch_size=args['training']['batch_size'],
    shuffle=True,
    num_workers=4,
)
val_loader = DataLoader(
    val_ds,
    batch_size=args['training']['batch_size'],
    shuffle=True,
    num_workers=4,
)

# -------------------------------------------------------------------------------
# loss and optimizer
# -------------------------------------------------------------------------------

if args['training']['loss'].lower() == 'mse':
    criterion = metrics.MSELoss()
elif args['training']['loss'].lower() == 'mae':
    criterion = metrics.MAELoss()
elif args['training']['loss'].lower() == 'ssim_mae':
    criterion = metrics.SSIMMAELoss(weights={'mae': 10., 'ssim': 1.})
else:
    raise ValueError('loss not listed')

optimizer = lambda mdl: torch.optim.Adam(mdl.parameters(), args['training']['lr'], betas=(0.9, 0.999))
scheduler = lambda opti: torch.optim.lr_scheduler.ExponentialLR(opti, args['training']['gamma'])

# metrics for logging
metrx = {
    'MSE': metrics.MSELoss(),
    'MAE': metrics.MAELoss(),
    'SSIM': metrics.MaskedSSIMLoss(),
}

# -------------------------------------------------------------------------------
# load model
# -------------------------------------------------------------------------------

mdl_args = args['model']
model_type = mdl_args['model'].lower()
if (model_type == 'unet'):
    import importlib
    model = importlib.import_module(f'src.unet.{model_type}')
    from src.unet import trainer

    mdl = model.UNet(
        n_channels=args['model']['input_channels'],
        n_classes=args['model']['n_time_points'],
        conv_size=args['model']['size_first_conv'],
    ).to(device)

    opti = optimizer(mdl)
    shdl = scheduler(opti)

    trnr = trainer.Trainer(
        model=mdl,
        metrics=metrx,
        batch_size=args['training']['batch_size'],
        device=device,
        criterion=criterion,
        optimizer=opti,
        scheduler=shdl,
        log_dir=log_dir,
    )
else:
    raise ValueError('unknown model')

# -------------------------------------------------------------------------------
# copy scripts to log_dir
# -------------------------------------------------------------------------------

log_files = [
    __file__,
    model.__file__,
    trainer.__file__,
    dataset.__file__,
    [pargs.config_file, 'config.yml'],
]
copy_files(log_files, log_dir)

# -------------------------------------------------------------------------------
# train model
# -------------------------------------------------------------------------------

trnr.train(train_loader, val_loader, args['training']['epochs'])
