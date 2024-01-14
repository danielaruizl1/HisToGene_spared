import os
import torch
import pytorch_lightning as pl
import argparse
import numpy as np
import wandb
import json
import torch_geometric
from pytorch_lightning.callbacks import ModelCheckpoint
from spared.datasets import HisToGeneDataset, get_dataset
from torch.utils.data import DataLoader
from vis_model import HisToGene
from utils_hist import *

# Add argparse
parser = argparse.ArgumentParser(description="Arguments for training HisToGene")
parser.add_argument("--dataset", type=str, default="10xgenomic_human_breast_cancer", help="Dataset to use")
parser.add_argument("--prediction_layer", type=str, default="c_d_log1p", help="Layer to use for prediction")
parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
tag = args.dataset
wandb.init(project="HisToGene", config=vars(args))

# Get datasets from the values defined in args
dataset = get_dataset(args.dataset)
dataset.adata.X = dataset.adata.layers[args.prediction_layer].astype(np.float32)
train_dataset = HisToGeneDataset(dataset.adata, "train")
val_dataset = HisToGeneDataset(dataset.adata, "val")

# Get dataloaders 
train_loader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, num_workers=4, shuffle=True)
n_pos = max(dataset.adata.obs["array_row"].max(), dataset.adata.obs["array_col"].max())+1
checkpoint_callback = ModelCheckpoint(dirpath="checkpoints/"+wandb.run.name, monitor='val_loss')
#FIXME: Change epochs to steps
trainer = pl.Trainer(devices=1, max_epochs=args.epochs, callbacks=[checkpoint_callback], logger = False)

# Get dataset config json
with open(os.path.join("spared","configs", f"{args.dataset}.json")) as json_file:
    dataset_config = json.load(json_file)

# Declare model
model = HisToGene(n_layers=8, n_genes=dataset_config["top_moran_genes"], 
                  learning_rate=args.lr, 
                  patch_size=dataset_config["patch_size"], 
                  n_pos=n_pos)
  
# Train model
trainer.fit(model, train_loader, val_loader)

#FIXME: Remove try except
try:
    test_dataset = HisToGeneDataset(dataset.adata, "test")
    test_loader =DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=True)
    trainer.test(model, dataloaders=test_loader, ckpt_path=checkpoint_callback.best_model_path)
except:
    pass

wandb.finish()
