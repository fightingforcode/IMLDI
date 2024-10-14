
import os

import numpy as np
import yaml
import argparse
from argparse import Namespace
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import torch
from torch.utils.data import DataLoader
from lime.lime_tabular import LimeTabularExplainer
from models.factory import create_model
from lib.metrics import *
from lib.dataset import MLDataset

torch.backends.cudnn.benchmark = True


class Evaluator(object):
    def __init__(self, cfg):
        super(Evaluator, self).__init__()
        dataset = MLDataset(cfg, cfg.test_path,is_train=False)
        self.dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)


        self.model = create_model(cfg.model, cfg=cfg)
        self.model.cuda()
        self.metrics = Metric(cfg.num_classes)
        #self.metrics = Metric(5)
        self.cfg = cfg


    @torch.no_grad()
    def run(self):
        model_dict = torch.load(self.cfg.ckpt_best_path)
        # print(model_dict.keys())
        if list(model_dict.keys())[0].startswith('module'):
            model_dict = {k[7:]: v for k, v in model_dict.items()}
        self.model.load_state_dict(model_dict)
        #print('loading best checkpoint success')

        self.model.eval()
        self.metrics.reset()
        motifs = []
        for batch in tqdm(self.dataloader):
            seq = batch['features'].cuda()
            targets = batch['labels'].cuda()
            ret = self.model(seq, batch['attention_mask'], batch['token_type_ids'], targets)
            # # [8,token]
            # motif = [m.split(' ') for m in batch['motif']]
            # sep_index = [m.index('[SEP]')-1 for m in motif]
            #
            # motif = np.array(motif)[:,1:] # [8,1058] del [CLS]
            #
            # # get attention score
            # # torch.topk()
            # attn_score = torch.sum(ret['att_token'],dim=1) # concat head
            # attn_score = torch.mean(attn_score[:,1:,1:],dim=1) # get each token score
            # # del [SEP]
            # row_indices = np.arange(attn_score.shape[0])
            # attn_score[row_indices,sep_index] = 0
            # # find max
            # index = torch.argmax(attn_score,dim=1).cpu().numpy()
            #
            # motif = motif[row_indices,index]
            # motifs.extend(motif)

            logit = ret['logits']
            scores = torch.sigmoid(logit).cpu().numpy()
            targets = targets.cpu().numpy()
            #self.metrics.update(scores[:,[5,0,6,1,7]],targets[:,[5,0,6,1,7]])
            self.metrics.update(scores, targets)
        # motifs = np.array([motif for motif in motifs if len(motif) >=3])
        # motifs = motifs.flatten()
        # np.savetxt(os.path.join(self.cfg.exp_dir, 'motif.txt'), motifs, fmt='%s')

        AP, f1_e, f1_micro, f1_macro, hamm_loss, acc, rloss, cover = self.metrics.main()
        #self.metrics.label_acc()
        self.metrics.plot_matrix()
        self.metrics.polt_prob()

        log_text = '\n' + '=' * 20 + ' Final Test Performance ' + '=' * 20 \
                   + '\n[ACC,\tAP,\tHammingloss,\tf1exam\tf1_micro,\tf1_macro,\trloss,\t cover]\n' + '{:.4f},\t{:4f},\t{:4f},\t{:4f},\t{:4f},\t{:4f},\t{:4f},\t{:4f}'.format(
            acc, AP, hamm_loss, f1_e, f1_micro, f1_macro, rloss, cover) \
                   + '\n' + '=' * 60

        print(log_text)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-dir', type=str, default='./exp10/')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()
    cfg_path = os.path.join(args.exp_dir, 'config.yaml')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError('config file not found in the {}!'.format(cfg_path))
    cfg = yaml.load(open(cfg_path, 'r'),Loader=yaml.FullLoader)
    cfg = Namespace(**cfg)
    cfg.ckpt_best_path = os.path.join(args.exp_dir, 'checkpoints', 'best_model.pth')
    cfg.batch_size = args.batch_size
    cfg.threshold = args.threshold
    print(cfg)

    evaluator = Evaluator(cfg)
    evaluator.run()
