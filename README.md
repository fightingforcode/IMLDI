### Introduction
# IMLDI
An interpretable multi-label mRNA subcellular localization predictor with the ability to handle label relevance and imbalance
### Data Preparation
1. Download mRNA dataset and DNAbert2_attention organize them as follow:
```
|dataset
|---- training_validation.fasta
|---- independent.fasta
```

2. Preprocess using following commands:
```bash
python scripts/mRNA.py
```

### Requirements
```
torch >= 1.12.0
```

### Training
One can use following commands to train model and reproduce the results reported in paper.
```bash
python train.py 
```
One can add `CUDA_VISIBLE_DEVICES=0,1,2,3` in front of the commands to enable distributed data parallel training with available GPUs.
### Evaluation

Pre-trained models are available in [link](https://pan.seu.edu.cn:443/link/524D2C7E5F89C0B2017AF5A746BD84BC). Download and put them in the `experiments` folder, then one can use following commands to reproduce results reported in paper.
```bash
python evaluate.py
```
