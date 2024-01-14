import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from vis_model import HisToGene
from HisToGene.utils_hist import *
from predict import model_predict, sr_predict
from dataset import ViT_HER2ST, ViT_SKIN

fold = 5
tag = '-htg_her2st_785_32_cv'
dataset = ViT_HER2ST(train=True, fold=fold)
train_loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=True)
model = HisToGene(n_layers=8, n_genes=785, learning_rate=1e-5)
trainer = pl.Trainer(gpus=0, max_epochs=100)
trainer.fit(model, train_loader)
trainer.save_checkpoint("model/last_train_"+tag+'_'+str(fold)+".ckpt")

fold = 5
tag = '-htg_her2st_785_32_cv'

model = HisToGene.load_from_checkpoint("model/last_train_"+tag+'_'+str(fold)+".ckpt",n_layers=8, n_genes=785, learning_rate=1e-5)
device = torch.device('cpu')
dataset = ViT_HER2ST(train=False,sr=False,fold=fold)
test_loader = DataLoader(dataset, batch_size=1, num_workers=4)
adata_pred, adata_truth = model_predict(model, test_loader, attention=False, device = device)
adata_pred = comp_tsne_km(adata_pred,4)

g = list(np.load('data/her_hvg_cut_1000.npy',allow_pickle=True))
adata_pred.var_names = g
sc.pp.scale(adata_pred)
adata_pred