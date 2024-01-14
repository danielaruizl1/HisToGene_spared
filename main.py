import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from vis_model import HisToGene
from utils_hist import *
from predict import model_predict, sr_predict
from dataset import ViT_HER2ST, ViT_SKIN
import argparse
import sys
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
sys.path.insert(0, '/media/SSD3/daruizl/ST')
from datasets import HisToGeneDataset
from utils import get_main_parser, get_dataset_from_args

# Training
parser_ST = get_main_parser()
args_ST = parser_ST.parse_args()
use_cuda = torch.cuda.is_available()
tag = args_ST.dataset
wandb.init(project="HisToGene", config=vars(args_ST))

# Get datasets from the values defined in args
dataset = get_dataset_from_args(args=args_ST)
dataset.adata.X = dataset.adata.layers[args_ST.prediction_layer].astype(np.float32)
train_dataset = HisToGeneDataset(dataset.adata, "train")
val_dataset = HisToGeneDataset(dataset.adata, "val")

# Get dataloaders 
train_loader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, num_workers=4, shuffle=True)
n_pos = max(dataset.adata.obs["array_row"].max(), dataset.adata.obs["array_col"].max())+1
trainer = pl.Trainer(gpus=1, max_epochs=args_ST.epochs, checkpoint_callback = True, callbacks=[ModelCheckpoint(dirpath="../../../SSD5/daruizl/checkpoints/"+wandb.run.name, monitor='val_loss')], logger = False)

# Declare model
model = HisToGene(n_layers=8, n_genes=args_ST.top_moran_genes, learning_rate=args_ST.lr, patch_size=args_ST.patch_size, n_pos=n_pos)
  

if args_ST.train == True:
    
    trainer.fit(model, train_loader, val_loader)

    if args_ST.dataset == "stnet_dataset":
        test_dataset = HisToGeneDataset(dataset.adata, "test")
        test_loader =DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=True)
        trainer.test(model, dataloaders=test_loader, ckpt_path="best")

else:

    if args_ST.dataset == "V1_Breast_Cancer_Block_A":
        trainer.test(model, dataloaders=val_loader, ckpt_path=os.path.join("checkpoints","best_visium","*.ckpt"))
    else:
        test_dataset = HisToGeneDataset(dataset.adata, "test")
        test_loader = torch_geometric.loader.DataLoader(datasets[f"test{args.dataset}"],batch_size=1)
        trainer.test(model, dataloaders=test_loader, ckpt_path=os.path.join("checkpoints","best_stnet","*.ckpt"))

wandb.finish()