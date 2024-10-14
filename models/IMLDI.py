import torch
import torch.nn as nn


from .factory import register_model,create_backbone
from lib.utils import get_loss_fn
from .utils import *
import numpy as np

__all__ = ['mlic']


class IMLDI(nn.Module):
    def __init__(self, backbone, feat_dim, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone

        self.text_feats = torch.tensor(np.load(cfg.embed_path), dtype=torch.float32).cuda()
        self.blocks = nn.ModuleList([Block(dim=feat_dim, num_heads=cfg.num_heads) for _ in range(cfg.depth)])

        self.depth = cfg.depth
        self.text_head_ = Head(feat_dim, cfg.num_classes)


        self.attention = LowRankBilinearAttention(feat_dim, feat_dim, feat_dim)

        self.criterion = get_loss_fn(cfg)

        self.net1 = nn.Sequential(
            nn.Conv1d(feat_dim, feat_dim, 1, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True)
        )
        self.net2 = nn.Sequential(
            nn.Conv1d(feat_dim, feat_dim, 1, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, att_mask,token_type_ids,y=None):
        seqfeat,att_token = self.backbone(x,att_mask,token_type_ids) # [8,token,768]
        # cls [8,768]

        tfeat = torch.stack([self.text_feats for _ in range(x.shape[0])], dim=0)

        for i in range(self.depth):
            tfeat, attn = self.blocks[i](seqfeat, tfeat)
            # tfeat [8,9,768]

        _alpha = self.attention(seqfeat, tfeat) # [8,9,token]
        f = self.net1(seqfeat.transpose(1, 2)).transpose(1, 2)
        _x = torch.bmm(_alpha, f)
        _x = self.net2(_x.transpose(1, 2)).transpose(1, 2) # [8,9,768]

        logits = self.text_head_(_x)
        # 定义一个池化层来将维度从 1071 缩小到 9
        #pooling = nn.AdaptiveAvgPool1d(9)

        # 对第二个维度进行池化
        #seqfeat_ = pooling(seqfeat.permute(0, 2, 1)).permute(0, 2, 1)
        #logits = self.text_head_(tfeat)
        loss = 0
        if self.training:
            loss = self.criterion(logits, y)

        return {
            'logits': logits,
            'attn_label': attn,  # [B,H,9,9]
            'loss': loss,
            'alpha': _alpha, # [B,9,token]
            'att_token':att_token,
        }



@register_model
def mlic(cfg):
    backbone, feat_dim = create_backbone(cfg.arch)
    model = IMLDI(backbone, feat_dim, cfg)
    return model


