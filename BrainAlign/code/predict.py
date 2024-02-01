# -- coding: utf-8 --
# @Time : 2022/10/16 9:11
# @Author : Biao Zhang
# @Email : littlebiao@outlook.com
# @File : predict.py
import numpy
import torch
from utils import load_data, set_params, evaluate
from module import HeCo
import warnings
import datetime
import pickle as pkl
import os
import random
from utils.logger import setup_logger
warnings.filterwarnings('ignore')
args = set_params()
if torch.cuda.is_available():
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device("cpu")

## name of intermediate document ##
own_str = args.dataset

## random seed ##
seed = args.seed
numpy.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def predict(save_path):

    logger = setup_logger("Build logging...", save_path, if_train=False)

    logger.info('Configs {}\n'.format(args))
    nei_index, feats, mps, pos, label, idx_train, idx_val, idx_test = \
        load_data(args.dataset, args.ratio, args.type_num)
    nb_classes = label.shape[-1]
    logger.info('number of classes = {}'.format(nb_classes))
    feats_dim_list = [i.shape[1] for i in feats]
    P = int(len(mps))
    logger.info("seed {}".format(args.seed))
    logger.info("Dataset: {}".format(args.dataset))
    logger.info("The number of meta-paths: {}".format(P))
    model = HeCo(args.hidden_dim, feats_dim_list, args.feat_drop, args.attn_drop,
                 P, args.sample_rate, args.nei_num, args.tau, args.lam)
    if torch.cuda.is_available():
        logger.info('Using CUDA')
        model.cuda()
        feats = [feat.cuda() for feat in feats]
        mps = [mp.cuda() for mp in mps]

    model.load_state_dict(torch.load(save_path + 'HeCo_' + own_str + '.pkl'))
    model.eval()
    # os.remove('HeCo_'+own_str+'.pkl')
    embeds = model.get_embeds(feats, mps)

    if args.save_emb:
        if not os.path.exists(args.save_path + "./embeds/" + args.dataset + "/"):
            os.makedirs(args.save_path + "./embeds/" + args.dataset + "/")
        f = open(args.save_path + "./embeds/" + args.dataset + "/" + str(args.turn) + ".pkl", "wb")
        pkl.dump(embeds.cpu().data.numpy(), f)
        f.close()

if __name__ == '__main__':
    save_path = '../data/mouse_human_sagittal/results/2022-10-14_11-40-11/'
    predict(save_path)