# -- coding: utf-8 --
# @Time : 2022/12/13 19:53
# @Author : Biao Zhang
# @Email : littlebiao@outlook.com
# @File : main_sr_rsc.py

import numpy
import torch
import sys

sys.path.append('../../')

import warnings
import datetime
import pickle as pkl
import os
import random
from BrainAlign.code.utils.logger import setup_logger

warnings.filterwarnings('ignore')
import sys
import argparse
import numpy as np
import torch
import time
seed = 268945
torch.autograd.set_detect_anomaly(True)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import argparse
from BrainAlign.SR_RSC.models import SubHIN
import logging

def parse_args():
    # input arguments
    parser = argparse.ArgumentParser(description='BiHIN')
    parser.add_argument('--gpu_num', nargs='?', default='0')
    parser.add_argument('--model', nargs='?', default='SubHIN')
    parser.add_argument('--dataset', nargs='?', default='dblp')
    parser.add_argument('--save_path', nargs='?', default='./results')

    parser.add_argument('--nb_epochs', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=50)

    #     parser.add_argument('--att_hid_units', type=int, default=64)
    parser.add_argument('--hid_units', type=int, default=256)  # 128 best for dblp and yelp, larger datasets
    parser.add_argument('--hid_units2', type=int, default=128)
    parser.add_argument('--out_ft', type=int, default=64)

    parser.add_argument('--drop_prob', type=float, default=0.0)
    parser.add_argument('--lamb', type=float, default=0.5,
                        help='coefficient for the losses in node task')
    parser.add_argument('--lamb_lp', type=float, default=1.0,
                        help='coefficient for the losses in link task')
    parser.add_argument('--margin', type=float, default=0.8,
                        help='coefficient for the margin loss')
    parser.add_argument('--isBias', action='store_true', default=False)
    parser.add_argument('--isAtt', action='store_true', default=False)
    parser.add_argument('--isLP', action='store_true', default=False)  # link prediction
    parser.add_argument('--isSemi', action='store_true', default=False)  # semi-supervised learning

    return parser.parse_known_args()


def printConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))
    print(args_names)
    print(args_vals)


def train(cfg, logger):
    args = cfg.SRRSC_args
    #args, unknown = parse_args()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    embedder = SubHIN(args)
    start = time.time()
    save_path = embedder.training(logger)
    print('time (s):%.2f' % (time.time() - start))

    return save_path


def run_srrsc(cfg, logger):
    # save_path = '../data/mouse_human_sagittal/results/'
    # dataset = sys.argv[1]
    # args = set_params(dataset)
    # args = cfg.SRRSC_args
    if cfg.SRRSC_args.device != 'cpu':
        device = torch.device("cuda:" + str(cfg.SRRSC_args.gpu_num))
        torch.cuda.set_device("cuda:" + str(cfg.SRRSC_args.gpu_num))
    else:
        device = torch.device("cpu")
        #torch.cuda.set_device("cpu")

    ## name of intermediate document ##
    # own_str = args.dataset

    ## random seed ##
    seed = cfg.SRRSC_args.seed
    numpy.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if cfg.SRRSC_args.dataset == 'mouse_human_binary':
        if cfg.SRRSC_args.if_pretrained == True:
            # args.if_pretrained = True
            # args.pretrained_model_path = "../data/mouse_human_binary/results/2022-11-30_17-18-47/"
            cfg.SRRSC_args.save_path = cfg.SRRSC_args.pretrained_model_path
        #logger = setup_logger("Build logging...", cfg.SRRSC_args.save_path, if_train=True)

        logger.info('Configs {}\n'.format(cfg.SRRSC_args))
        save_path = train(cfg, logger)
    elif cfg.SRRSC_args.dataset == 'mouse_human_four':
        if cfg.SRRSC_args.if_pretrained == True:
            # args.if_pretrained = True
            # args.pretrained_model_path = "../data/mouse_human_binary/results/2022-11-30_17-18-47/"
            cfg.SRRSC_args.save_path = cfg.SRRSC_args.pretrained_model_path
        #logger = setup_logger("Build logging...", cfg.SRRSC_args.save_path, if_train=True)

        logger.info('Configs {}\n'.format(cfg.SRRSC_args))
        save_path = train(cfg, logger)
    else:
        #logger = setup_logger("Build logging...", cfg.SRRSC_args.save_path, if_train=True)
        if cfg.SRRSC_args.if_pretrained == True:
            # args.if_pretrained = True
            # args.pretrained_model_path = "../data/mouse_human_binary/results/2022-11-30_17-18-47/"
            cfg.SRRSC_args.save_path = cfg.SRRSC_args.pretrained_model_path
        logger.info('Configs {}\n'.format(cfg.SRRSC_args))
        save_path = train(cfg, logger)


