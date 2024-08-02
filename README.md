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

export PYTHONPATH=/anaconda3/envs/histogene_spared/lib/python3.10/site-packages/spared:$PYTHONPATH
```

## Train HisToGene

To use a dataset from spared, train HisToGene by running:
```bash
python main_histogene_spared.py --dataset spared_dataset_name
```

To use a dataset not included in spared, train HisToGene by running:
```bash
cd v2
python main_histogene_spared.py --dataset adata_path
```
`adata_path` should be the path to an adata.h5ad with the same structure than the used for SpaRED adata files.

