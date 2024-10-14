# IMLDI
An interpretable multi-label mRNA subcellular localization predictor with the ability to handle label relevance and imbalance
### Data Preparation
1. Download mRNA dataset and DNAbert2_attention in [link](https://pan.seu.edu.cn:443/link/524D2C7E5F89C0B2017AF5A746BD84BC) organize them as follow:
```
|dataset
|---- training_validation.fasta
|---- independent.fasta
|DNAbert2_attention
|---- bert_layers.py
|---- bert_padding.py
|----pytorch_model.bin
|---- ...
```

2. Preprocess using following commands:
```bash
python scripts/mRNA.py
python scripts/embedding.py
```

### Requirements
```
torch >= 1.12.0
transformers=4.41.2
scikit-learn=1.5.0
captum=0.7.0
biopython=1.83
```

### Training
One can use following commands to train model and reproduce the results reported in paper.
```bash
python train.py 
```
One can add `CUDA_VISIBLE_DEVICES=0,1,2,3` in front of the commands to enable distributed data parallel training with available GPUs.
### Five-folds evaluation
```bash 
python lib/data_crossval.py
bash run_crossval.sh
```
### Evaluation

Pre-trained models are available in [link](https://pan.seu.edu.cn:443/link/524D2C7E5F89C0B2017AF5A746BD84BC).
```bash
python evaluate.py
```
### Prediction pipline
Suppose you want to indentfy the subcellular localization of mRNA sequences, you could
```bash
python pipeline.py 
```
