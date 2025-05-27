import os
import torch
import pytorch_lightning as pl
import argparse
import numpy as np
import wandb
import json
import torch_geometric
from pytorch_lightning.callbacks import ModelCheckpoint
from spared.spared_datasets import HisToGeneDataset, get_dataset
from spared.denoising import spackle_cleaner
from torch.utils.data import DataLoader
from vis_model import HisToGene
from utils_hist import *
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime
from anndata.experimental.pytorch import AnnLoader
import pandas as pd
import sys

# Auxiliary function to use booleans in parser
str2bool = lambda x: (str(x).lower() == 'true')
str2intlist = lambda x: [int(i) for i in x.split(',')]
str2floatlist = lambda x: [float(i) for i in x.split(',')]
str2h_list = lambda x: [str2intlist(i) for i in x.split('//')[1:]]

# FIXME: This argparser is not working because of get_dataset
# Add argparse
parser = argparse.ArgumentParser(description="Arguments for training HisToGene")
parser.add_argument("--dataset", type=str, default="10xgenomic_human_breast_cancer", help="Dataset to use")
parser.add_argument("--prediction_layer", type=str, default="c_t_log1p", help="Layer to use for prediction")
parser.add_argument('--lr', type=float, default=0.004642, help='Learning rate')
parser.add_argument('--use_optimal_lr', type=str2bool, default=False, help='Whether or not to use the optimal learning rate in csv for the dataset.')
parser.add_argument('--max_steps', type=int, default=2000, help='Number of iterations'),
parser.add_argument('--val_check_interval', type=int, default=10, help='Number of iterations between validation check'),
parser.add_argument('--noisy_training', type=str2bool, default=False, help='Whether or not to do noisy training'),
parser.add_argument('--opt_metric', type=str, default="MSE", help='Metric to optimize')
parser.add_argument("--original_index", type=str2bool, default=False, help="Whether to use the original index")
args = parser.parse_args()

# Declare device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# If exp_name is None then generate one with the current time
args.exp_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# Start wandb configs
wandb_logger = WandbLogger(
    project="histogene_spared",
    name=args.exp_name,
    log_model=False,
    config=vars(args),
)

# Get datasets from the values defined in args
dataset = get_dataset(args.dataset, visualize=False)
if args.original_index:
    dataset.adata = dataset.adata[:, np.argsort(dataset.adata.var['original_index'].astype(int))].copy()

# Denoise the dataset if necessary
if (args.prediction_layer == 'c_t_log1p') and (not args.prediction_layer in dataset.adata.layers):
    dataset.adata, _  = spackle_cleaner(adata=dataset.adata, dataset=args.dataset, from_layer="c_d_log1p", to_layer="c_t_log1p", device=device)
    # Replace current adata.h5ad file for the one with the completed data layer.
    dataset.adata.write_h5ad(os.path.join(dataset.dataset_path, "adata.h5ad"))

elif (args.prediction_layer == "c_t_deltas") and (not args.prediction_layer in dataset.adata.layers):
    dataset.adata, _  = spackle_cleaner(adata=dataset.adata, dataset=args.dataset, from_layer="c_d_deltas", to_layer="c_t_deltas", device=device)
    # Replace current adata.h5ad file for the one with the completed data layer.
    dataset.adata.write_h5ad(os.path.join(dataset.dataset_path, "adata.h5ad"))

if args.noisy_training:    
    # Copy the layer c_d_log1p to the layer noisy
    c_d_log1p = dataset.adata.layers['c_d_log1p'].astype(np.float32).copy()
    # Get zero mask
    zero_mask = ~dataset.adata.layers['mask']
    # Zero out the missing values
    c_d_log1p[zero_mask] = 0
    # Add the layer to the adata
    dataset.adata.X = c_d_log1p
    # Give warning to say that the noisy layer is being used
    print('Using noisy layer for training. This will probably yield bad results.')
else:
    dataset.adata.X = dataset.adata.layers[args.prediction_layer].astype(np.float32)

train_dataset = HisToGeneDataset(dataset.adata, "train")
val_dataset = HisToGeneDataset(dataset.adata, "val")

# Get save path and create is in case it is necessary
save_path = os.path.join('checkpoints', args.exp_name)
os.makedirs(save_path, exist_ok=True)
# Save script arguments in json file
with open(os.path.join(save_path, 'script_params.json'), 'w') as f:
    json.dump(vars(args), f, indent=4)

# Obtain optimal lr depending on the dataset
if args.use_optimal_lr:
    optimal_models_directory_path =  '/home/daruizl/HisToGene_SEPAL/wandb_runs_csv/optimal_models_lr_ctlog1p.csv'
    optimal_lr_df = pd.read_csv(optimal_models_directory_path, sep=";")
    optimal_lr = float(optimal_lr_df[optimal_lr_df['Dataset'] == args.dataset]['histogene'])
    args.lr = optimal_lr
    print(f'Optimal lr for {args.dataset} is {optimal_lr}')

# Get dataloaders 
train_loader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, num_workers=4, shuffle=True)
n_pos = max(dataset.adata.obs["array_row"].max(), dataset.adata.obs["array_col"].max())+1
checkpoint_callback = ModelCheckpoint(dirpath=save_path, monitor='val_loss')

# Define the trainier and fit the model
trainer = pl.Trainer(
    max_steps=args.max_steps,
    val_check_interval=args.val_check_interval,
    log_every_n_steps=args.val_check_interval,
    check_val_every_n_epoch=None,
    devices=1,
    callbacks=[checkpoint_callback],
    enable_progress_bar=True,
    enable_model_summary=True,
    logger=wandb_logger
)

# Get dataset config json
spared_path = next((path for path in sys.path if 'spared' in path), None)
dataset_config_path = os.path.join(spared_path,"configs",args.dataset+".json")
with open(dataset_config_path) as json_file:
    dataset_config = json.load(json_file)

# Declare model
model = HisToGene(n_layers=8, n_genes=dataset_config["top_moran_genes"], 
                  learning_rate=args.lr, 
                  patch_size=dataset_config["patch_size"], 
                  n_pos=n_pos,
                  opt_metric=args.opt_metric)
  
# Train model
trainer.fit(model, train_loader, val_loader)

checkpoint_path= checkpoint_callback.best_model_path    

if dataset.adata.obs.split.nunique() > 2:
    test_dataset = HisToGeneDataset(dataset.adata, "test")
    test_loader =DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=True)
    trainer.test(model, dataloaders=test_loader, ckpt_path=checkpoint_path)

# Visualizations

# Load the best model after training
checkpoint = torch.load(checkpoint_path)

# Load the model
model = HisToGene(n_layers=8, n_genes=dataset_config["top_moran_genes"], 
                  learning_rate=args.lr, 
                  patch_size=dataset_config["patch_size"], 
                  n_pos=n_pos,
                  opt_metric=args.opt_metric)

model.load_state_dict(checkpoint['state_dict'])

def get_predictions(model)->None:

    # Get complete Histogene dataset
    complete_dataset = HisToGeneDataset(dataset.adata, None)

    # Get complete dataloader
    dataloader = DataLoader(complete_dataset, batch_size=1, num_workers=4, shuffle=False)

    # Define global variables
    glob_expression_pred = None
    glob_ids = None

    # Set model to eval mode
    model.eval()

    # Get complete predictions
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            expression_pred = model(data[0].to(device),data[1].to(device))
            expression_pred = expression_pred.squeeze(0)
            # Concat batch to get global predictions and IDs
            glob_expression_pred = expression_pred if glob_expression_pred is None else torch.cat((glob_expression_pred, expression_pred))
             # Get the slide from the index
            slide = complete_dataset.idx_2_slide[idx]
            # Get the adata of the slide
            adata_slide = complete_dataset.adata[complete_dataset.adata.obs.slide_id == slide]
            glob_ids = adata_slide.obs['unique_id'].tolist() if glob_ids is None else glob_ids + adata_slide.obs['unique_id'].tolist()

        # Handle delta prediction
        if 'deltas' in args.prediction_layer:
            mean_key = f'{args.prediction_layer}_avg_exp'.replace('deltas', 'log1p')
            means = torch.tensor(dataset.adata.var[mean_key], device=glob_expression_pred.device)
            glob_expression_pred = glob_expression_pred+means
        
        # Put complete predictions in a single dataframe
        pred_matrix = glob_expression_pred.detach().cpu().numpy()
        pred_df = pd.DataFrame(pred_matrix, index=glob_ids, columns=dataset.adata.var_names)
        pred_df = pred_df.reindex(dataset.adata.obs.index)

        # Log predictions to wandb
        wandb_df = pred_df.reset_index(names='sample')
        wandb.log({'predictions': wandb.Table(dataframe=wandb_df)})
        
        # Add layer to adata
        dataset.adata.layers[f'predictions,{args.prediction_layer}'] = pred_df

# Get global prediction layer 
get_predictions(model.to(device))

# Get log final artifacts
#dataset.log_pred_image()

wandb.finish()
