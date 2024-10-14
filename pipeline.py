# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2022-11-11
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2022 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os

import numpy as np
import yaml
import argparse
from argparse import Namespace
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
import pandas as pd
warnings.filterwarnings("ignore")
import torch
import torch.nn.functional as F
from models.factory import create_model
from lib.metrics import *


torch.backends.cudnn.benchmark = True
path = '/tmp/pycharm_project_850/DNAbert2_attention'

class Pipeline(object):
    def __init__(self, cfg):
        super(Pipeline, self).__init__()
        self.model = create_model(cfg.model, cfg=cfg)
        self.model.cuda()
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.labels = [line.strip() for line in open(cfg.label_path)]
    def truncate_sequence(self,sequence, max_length):
        if len(sequence) <= max_length:
            return sequence
        else:
            return sequence[-max_length:]

    def get_attention(self, attn, mask,token):
        plt.figure(figsize=(10, 4))
        attn = attn.squeeze(0)
        mask = mask.squeeze(0).unsqueeze(0).repeat(9,1)
        no_padindex = mask.bool()
        attn = attn[no_padindex].view(9,-1)[:,1:-1] # del sep and cls
        attn,_ = torch.sort(attn,descending=True)
        #attn = F.softmax(attn[:,:10],dim=-1)
        attn = F.normalize(attn[:,:10],p=2,dim=0)
        df = pd.DataFrame(attn.cpu().numpy(),columns=token[:10],index=self.labels)
        sns.heatmap(df,annot=True)
        plt.show()

    @torch.no_grad()
    def run(self):
        model_dict = torch.load(self.cfg.ckpt_best_path)
        # print(model_dict.keys())
        if list(model_dict.keys())[0].startswith('module'):
            model_dict = {k[7:]: v for k, v in model_dict.items()}
        self.model.load_state_dict(model_dict)
        print('loading best checkpoint success')

        with open('./input.fasta', 'r') as fr:
            for line in fr.readlines():
                print('==========================================')
                if line.startswith('>'):
                    continue
                seq = line.strip()
                seq = self.truncate_sequence(seq, 5000)
                if seq:
                    inputs = self.tokenizer(
                            seq,
                            add_special_tokens=True,
                            max_length=1000,
                            padding='max_length',
                            return_token_type_ids=True,
                            truncation=True,
                            return_attention_mask=True,
                            return_tensors='pt'
                        )
                    # 推断（预测）
                    with torch.no_grad():
                        self.model.eval()
                        token = self.tokenizer.decode(inputs['input_ids'].flatten(),skip_special_tokens=True).split(' ')
                        outputs = self.model(inputs["input_ids"].cuda(), inputs["attention_mask"].cuda(), inputs["token_type_ids"].cuda())
                        scores = torch.sigmoid(outputs['logits']).cpu().numpy()[0]
                        plt.figure(figsize=(15, 6))

                        label_attn = outputs['attn_label'].mean(dim=1).squeeze(0)
                        mask = np.triu(np.ones_like(label_attn.cpu().numpy(), dtype=bool))
                        label_attn = F.normalize(label_attn,p=2,dim=0)
                        df = pd.DataFrame(label_attn.cpu().numpy(),index=self.labels,columns=self.labels)
                        sns.heatmap(df, annot=True,mask=mask,cmap='coolwarm')
                        plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')
                        # attn = outputs['alpha']
                        # self.get_attention(attn,inputs['attention_mask'],token)
                        for i, score in enumerate(scores):
                                print(self.labels[i], score)
                print('==========================================')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-dir', type=str, default='./exp10/')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--input-path', type=str, default='./test.fasta')

    args = parser.parse_args()
    cfg_path = os.path.join(args.exp_dir, 'config.yaml')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError('config file not found in the {}!'.format(cfg_path))
    cfg = yaml.load(open(cfg_path, 'r'),Loader=yaml.FullLoader)
    cfg = Namespace(**cfg)
    cfg.ckpt_best_path = os.path.join(args.exp_dir, 'checkpoints','best_model.pth')
    cfg.threshold = args.threshold
    print(cfg)

    evaluator = Pipeline(cfg)
    evaluator.run()