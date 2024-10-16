# RNALocAtt
An enhanced imbalanced multi-label predictor for mRNA subcellular localization based on attention mechanisms
### Data Preparation
1. Download mRNA dataset and DNAbert2_attention in [link](https://drive.google.com/drive/folders/1D-L1-kJcjiAl4lYxrvl6yNF05CjhsClP) organize them as follow:
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

Pre-trained models named exp10 are available in [link](https://drive.google.com/drive/folders/1D-L1-kJcjiAl4lYxrvl6yNF05CjhsClP).
```bash
python evaluate.py
```
### Prediction pipline
Suppose you want to indentfy the subcellular localization of mRNA sequences, you could
```bash
python pipeline.py 
```
