import os
from Bio import SeqIO
from collections import defaultdict
from tqdm import tqdm

# def seq_pad_trunc(data):
#     length = len(data)
#     if length > 6000:
#         processed_data = data[:3000] + data[-3000:]
#     else:
#         processed_data = data + (6000 - length)*'N'
#
#     return processed_data

data_dir = '../dataset/'
save_dir = '../data/mRNA'
label_list = ["Exosome","Nucleus","Nucleoplasm","Chromatin","Nucleolus","Cytosol","Membrane","Ribosome","Cytoplasm"]

if not os.path.exists(save_dir):
    os.mkdir(save_dir)


train_path = os.path.join(data_dir, 'training_validation.fasta')
test_path = os.path.join(data_dir, 'independent.fasta')

train_info = list(SeqIO.parse(train_path, "fasta"))
test_info = list(SeqIO.parse(test_path, "fasta"))

train_data, test_data = [], []
for info in train_info:
    label = info.id.split(',')[0].split('|')
    # seq = seq_pad_trunc(info.seq)
    train_data.append('{}\t{}\n'.format(info.seq, ','.join(label)))

for info in test_info:
    label = info.id.split(',')[0].split('|')
    # seq = seq_pad_trunc(info.seq) pad
    test_data.append('{}\t{}\n'.format(info.seq, ','.join(label)))

with open(os.path.join(save_dir, 'train.txt'), 'w') as fw:
    fw.writelines(train_data)
with open(os.path.join(save_dir, 'test.txt'), 'w') as fw:
    fw.writelines(test_data)
with open(os.path.join(save_dir, 'label.txt'), 'w') as fw:
    fw.writelines(['{}\n'.format(t) for t in label_list])

print('all seq number: ', len(train_data) + len(test_data))
print('train seq number: ', len(train_data))
print('test seq number: ', len(test_data))

