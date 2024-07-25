## Set up

Run the following to define your environment in terminal:

```bash
conda create -n histogene_spared
conda activate histogene_spared
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install lightning -c conda-forge
pip install torch-geometric==2.3.1
pip install squidpy
pip install wandb
pip install wget
pip install combat
pip install opencv-python
pip install positional-encodings[pytorch]
pip install plotly
pip install sh
pip install einops

```

# Usage
```python
import torch
from vis_model import HisToGene

model = HisToGene(
    n_genes=1000, 
    patch_size=112, 
    n_layers=4, 
    dim=1024, 
    learning_rate=1e-5, 
    dropout=0.1, 
    n_pos=64
)

# flatten_patches: [N, 3*W*H]
# coordinates: [N, 2]

pred_expression = model(flatten_patches, coordinates)  # [N, n_genes]

```

## System environment
Required package:
- PyTorch >= 1.8
- pytorch-lightning >= 1.4
- scanpy >= 1.8

## Parameters
- `n_genes`: int.  
  Amount of genes.
- `patch_size`: int.  
  Width/diameter of the spots.
- `n_layers`: int, default `4`.  
  Number of Transformer blocks.
- `dim`: int.  
  Dimension of the embeddings.
- `learning_rate`: float between `[0, 1]`, default `1e-5`.  
  Learning rate.
- `dropout`: float between `[0, 1]`, default `0.1`.  
  Dropout rate in the Transformer.
- `n_pos`: int, default `64`.  
   Maximum number of the coordinates.

# HisToGene pipeline
See [tutorial.ipynb](tutorial.ipynb)

# References
https://github.com/almaan/her2st
