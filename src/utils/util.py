import logging
import os.path
import math
import random
import time

import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from transformers import AdamW
from transformers import get_polynomial_decay_schedule_with_warmup

import torch.distributed as dist




def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    # prepare distributed
    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    # set cuda device
    torch.cuda.set_device(args.gpu)
    
    
def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    # prepare distributed
    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    # set cuda device
    torch.cuda.set_device(args.gpu)
    
    
def setup_device(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()
    # args.n_gpu = 1


def setup_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed) 


def setup_logging(args):
    time_ = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    log_file = os.path.join(args.savedmodel_path, f'{args.model_type}-{time_}.txt')
    logging.basicConfig(filename=log_file, format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)

    logger = logging.getLogger(__name__)
    return logger



def build_optimizer(args, model, t_total, T_mult=1, rewarm_epoch_num=1):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    no_decay_param_tp = [(n, p) for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
    decay_param_tp = [(n, p) for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]

    no_decay_bert_param_tp = [(n, p) for n, p in no_decay_param_tp if "bert." in n]
    no_decay_nobert_param_tp = [(n, p) for n, p in no_decay_param_tp if "bert." not in n]

    decay_bert_param_tp = [(n, p) for n, p in decay_param_tp if "bert." in n]
    decay_nobert_param_tp = [(n, p) for n, p in decay_param_tp if "bert." not in n]


    optimizer_grouped_parameters = [  # 分层设置学习率
        {'params': [p for n, p in no_decay_bert_param_tp], 'weight_decay': 0.01, 'lr': args.learning_rate},
        {'params': [p for n, p in decay_bert_param_tp], 'weight_decay': 0.0, 'lr': args.learning_rate},

        {'params': [p for n, p in no_decay_nobert_param_tp], 'weight_decay': 0.01, 'lr': args.learning_rate},
        {'params': [p for n, p in decay_nobert_param_tp], 'weight_decay': 0.0, 'lr': args.learning_rate}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, weight_decay=0.0005)

    total_steps = t_total
    WARMUP_RATIO = 0.1
    warmup_steps = int(WARMUP_RATIO * total_steps)
    print('total steps: ', t_total)
    scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=1000,
            num_training_steps=t_total,
            lr_end=0,
            power=1,
        )
    
    return optimizer, scheduler



