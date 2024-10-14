# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2021-8-9
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2021 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import sys
import time
import random
import traceback
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lib.utils import *
from lib.metrics import *
from lib.dataset import MLDataset
from models.factory import create_model
torch.backends.cudnn.benchmark = True
from tqdm import tqdm
logger = logging.getLogger(__name__)

class Trainer(object):
    def __init__(self, cfg, world_size, rank):
        super(Trainer, self).__init__()
        self.distributed = world_size > 1
        batch_size = cfg.batch_size // world_size if self.distributed else cfg.batch_size
        if cfg.mode == 'cross_val':
            train_path = cfg.fold_train_path
            test_path = cfg.fold_test_path
        else:
            train_path = cfg.train_path
            test_path = cfg.test_path
        train_dataset = MLDataset(cfg,train_path)
        val_dataset = MLDataset(cfg,test_path)
        if self.distributed:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        else:
            self.train_sampler = val_sampler = None
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(self.train_sampler is None),
                                        sampler=self.train_sampler)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,sampler=val_sampler)

        torch.cuda.set_device(rank)
        self.model = create_model(cfg.model, cfg=cfg)
        self.model.cuda(rank)

        self.ema_model = ModelEma(self.model, decay=cfg.ema_decay)
        if self.distributed:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[rank], find_unused_parameters=True)

        parameters = self.model.parameters()
        self.optimizer = get_optimizer(parameters, cfg)
        self.lr_scheduler = get_lr_scheduler(self.optimizer, cfg, steps_per_epoch=len(self.train_loader))
        # self.warmup_initial_lr = warmup_initial_lr
        # for g in self.optimizer.param_groups:
        #     g['lr'] = self.warmup_initial_lr
        self.warmup_scheduler = WarmUpLR(self.optimizer, len(self.train_loader) * cfg.warmup_epoch)

        self.criterion = get_loss_fn(cfg)
        self.early_stopping = EarlyStopping(patience=4)
        self.metrics = Metric(cfg.num_classes)
        self.metrics_ema = Metric(cfg.num_classes)
        self.cfg = cfg
        self.best_acc = 0
        self.global_step = 0
        self.notdist_or_rank0 = (not self.distributed) or (self.distributed and rank == 0)
        if self.notdist_or_rank0:
            self.logger = get_logger(cfg.log_path, __name__)
            self.writer = SummaryWriter(log_dir=cfg.exp_dir)

    def run(self):
        for epoch in range(self.cfg.max_epochs):
            if self.distributed:
                self.train_sampler.set_epoch(epoch)
            self.train(epoch)
            acc = self.validation(epoch)
            self.lr_scheduler.step(acc)
            is_save, is_terminate = self.early_stopping(acc)
            if is_terminate:
                break
            if is_save:
                torch.save(self.ema_model.state_dict(), self.cfg.ckpt_best_path)

        if self.notdist_or_rank0:
            self.logger.info('\ntraining over, best validation score: {} acc'.format(self.early_stopping.best_score))

    def train(self, epoch):
        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.amp)
        self.model.train()

        loop = tqdm(self.train_loader, desc="training")
        for batch in loop:
            batch_begin = time.time()
            seq = batch['features'].cuda()
            targets = batch['labels'].cuda()
            with torch.cuda.amp.autocast(enabled=self.cfg.amp):
                ret = self.model(seq, batch['attention_mask'],batch['token_type_ids'],y=targets)

            loss = ret['loss']
            self.optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            dur = time.time() - batch_begin
            loop.set_postfix({'loss': '{0:1.5f}'.format(loss)})
            #self.lr_scheduler.step()
            self.ema_model.update(self.model)

            if self.global_step % (len(self.train_loader) // 6) == 0 and self.notdist_or_rank0:
                lr = get_lr(self.optimizer)
                self.writer.add_scalar('Loss/train', loss, self.global_step)
                self.writer.add_scalar('lr', lr, self.global_step)
                self.logger.info('TRAIN [epoch {}] loss: {:4f}  lr:{:.6f} time:{:.4f}'
                                 .format(epoch, loss, lr, dur))
            if epoch < self.cfg.warmup_epoch:
                self.warmup_scheduler.step()

            self.global_step += 1

    @torch.no_grad()
    def validation(self, epoch):
        self.model.eval()
        self.ema_model.eval()
        self.metrics.reset()
        self.metrics_ema.reset()
        loop = tqdm(self.val_loader, desc="valing")
        for batch in loop:
            seq = batch['features'].cuda()

            targets = batch['labels'].cuda()

            logits = self.model(seq,batch['attention_mask'],batch['token_type_ids'])['logits']
            scores = torch.sigmoid(logits)
            logits = self.ema_model(seq,batch['attention_mask'],batch['token_type_ids'])['logits']
            ema_scores = torch.sigmoid(logits)
            if self.distributed:
                scores = concat_all_gather(scores)
                ema_scores = concat_all_gather(ema_scores)
                targets = concat_all_gather(targets)

            targets = targets.cpu().numpy()
            scores = scores.detach().cpu().numpy()
            self.metrics.update(scores, targets)
            ema_scores = ema_scores.detach().cpu().numpy()
            self.metrics_ema.update(ema_scores, targets)

        if self.distributed:
            dist.barrier()

        AP,f1_e,f1_micro,f1_macro,hamm_loss,acc,rloss,cover = self.metrics.main()
        ema_AP,ema_f1_e,ema_f1_micro,ema_f1_macro,ema_hamm_loss,ema_acc,ema_rloss,ema_cover = self.metrics_ema.main()

        log_text = '\n' + '=' * 20 + ' Final Test Performance ' + '=' * 20 \
                   + '\n[ACC,\tAP,\tHammingloss,\tf1exam\tf1_micro,\tf1_macro,\trloss,\t cover]\n' + '{:.4f},\t{:4f},\t{:4f},\t{:4f},\t{:4f},\t{:4f},\t{:4f},\t{:4f}'.format(
            acc, AP, hamm_loss,f1_e,f1_micro, f1_macro,rloss,cover) \
                   + '\n' + '=' * 60

        log_text_ema = '\n' + '=' * 20 + ' Final Test Performance ' + '=' * 20 \
                   + '\n[ACC_ema,\tAP_ema,\tHammingloss_ema,\tf1exam_ema\tf1_micro_ema,\tf1_macro_ema,\trloss_ema,\t cover_ema]\n' + '{:.4f},\t{:4f},\t{:4f},\t{:4f},\t{:4f},\t{:4f},\t{:4f},\t{:4f}'.format(
            ema_acc, ema_AP, ema_hamm_loss, ema_f1_e, ema_f1_micro, ema_f1_macro,ema_rloss,ema_cover) \
                   + '\n' + '=' * 60
        if self.notdist_or_rank0:
            self.writer.add_scalar('acc/val', acc, self.global_step)
            self.writer.add_scalar('ema_acc/val', ema_acc, self.global_step)

            self.logger.info("VALID [epoch {}] acc: {:.4f} ema_acc: {:.4f} best acc: {:.4f}"
                             .format(epoch, acc, ema_acc, max(max(ema_acc,acc), self.best_acc)))
            self.best_acc = max(max(ema_acc, acc), self.best_acc)
            self.logger.info("Validation [epoch {}]\n{}".format(epoch, log_text))
            self.logger.info("Validation [epoch {}] EMA\n{}".format(epoch, log_text_ema))

        return max(acc, ema_acc)


def main_worker(local_rank, ngpus_per_node, cfg, port=None):
    world_size = ngpus_per_node  # only single node is enough.
    if ngpus_per_node > 1:
        init_method = 'tcp://127.0.0.1:{}'.format(port)
        dist.init_process_group(backend='nccl', init_method=init_method, world_size=world_size, rank=local_rank)
    trainer = Trainer(cfg, world_size, local_rank)
    trainer.run()


if __name__ == "__main__":
    args = get_args()
    cfg = prepare_env(args, sys.argv)

    try:
        ngpus_per_node = torch.cuda.device_count()
        if ngpus_per_node > 1:
            port = get_port()
            setup_seed(cfg.seed)
            mp.spawn(main_worker, args=(ngpus_per_node, cfg, port,), nprocs=ngpus_per_node)
        else:
            setup_seed(cfg.seed)
            main_worker(0, ngpus_per_node, cfg)
    except (Exception, KeyboardInterrupt):
        print(traceback.format_exc())
        if not os.path.exists(cfg.ckpt_ema_best_path):
            clear_exp(cfg.exp_dir)
