import numpy
import torch
import sys
sys.path.append('../../')
from BrainAlign.code.utils import load_data, set_params, evaluate
from BrainAlign.code.module import HeCo
import warnings
import datetime
import pickle as pkl
import os
import random
from BrainAlign.code.utils.logger import setup_logger
warnings.filterwarnings('ignore')
import sys
import argparse

def train(cfg, logger):

    args = cfg.HECO_args
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)


    nei_index, feats, mps, pos, label, idx_train, idx_val, idx_test = \
        load_data(cfg, args.dataset, args.ratio, args.type_num, args.target_node)
    nb_classes = label.shape[-1]
    logger.info('number of classes = {}'.format(nb_classes))
    feats_dim_list = [i.shape[1] for i in feats]
    P = int(len(mps))
    logger.info("seed {}".format(args.seed))
    logger.info("Dataset: {}".format(args.dataset))
    logger.info("The number of meta-paths: {}".format(P))
    
    model = HeCo(args.hidden_dim, feats_dim_list, args.feat_drop, args.attn_drop,
                    P, args.sample_rate, args.nei_num, args.tau, args.lam)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)

    if torch.cuda.is_available():
        logger.info('Using CUDA')
        model.cuda()
        feats = [feat.cuda() for feat in feats]
        mps = [mp.cuda() for mp in mps]
        pos = pos.cuda()
        label = label.cuda()
        idx_train = [i.cuda() for i in idx_train]
        idx_val = [i.cuda() for i in idx_val]
        idx_test = [i.cuda() for i in idx_test]

    cnt_wait = 0
    best = 1e9
    best_t = 0

    if args.if_pretrained == True:
        logger.info('Loading {}th epoch'.format(best_t))
        model.load_state_dict(torch.load(args.pretrained_model_path + 'HeCo_' + args.target_node + '_' + args.dataset+'.pkl'))
        '''
        model.eval()
        # os.remove('HeCo_'+own_str+'.pkl')
        embeds = model.get_embeds(feats, mps)
        if not os.path.exists(args.save_path + "/embeds/" + "/"):
            os.makedirs(args.save_path + "/embeds/" + "/")
        f = open(args.save_path+"/embeds/"+"/"+"node_"+args.target_node+".pkl", "wb")
        pkl.dump(embeds.cpu().data.numpy(), f)
        f.close()
        return 0
        '''

    starttime = datetime.datetime.now()
    for epoch in range(args.nb_epochs):
        model.train()
        optimiser.zero_grad()
        loss = model(feats, pos, mps, nei_index)
        logger.info("Epoch {}, loss = {}".format(epoch, loss.data.cpu().numpy()))
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), args.save_path + 'HeCo_' + args.target_node + '_' + args.dataset+'.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            logger.info('Early stopping!')
            break
        loss.backward()
        optimiser.step()
        
    logger.info('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load(args.save_path + 'HeCo_' + args.target_node + '_' + args.dataset+'.pkl'))
    model.eval()
    #os.remove('HeCo_'+own_str+'.pkl')
    embeds = model.get_embeds(feats, mps)

    if args.save_emb:
        if not os.path.exists(args.save_path+"/embeds/"):
            os.makedirs(args.save_path+"/embeds/")
        f = open(args.save_path+"/embeds/"+"node_"+args.target_node+".pkl", "wb")
        pkl.dump(embeds.cpu().data.numpy(), f)
        f.close()

    '''
    for i in range(len(idx_train)):
        evaluate(embeds[0], args.ratio[i], idx_train[i], idx_val[i], idx_test[i], label, nb_classes, device, args.dataset,
                 args.eva_lr, args.eva_wd)
    '''
    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    logger.info("Total time: {}".format(time))
    
    return args.save_path


def run_heco(cfg):
    #save_path = '../data/mouse_human_sagittal/results/'
    #dataset = sys.argv[1]
    #args = set_params(dataset)
    #args = cfg.HECO_args
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(cfg.HECO_args.gpu))
        torch.cuda.set_device(cfg.HECO_args.gpu)
    else:
        device = torch.device("cpu")

    ## name of intermediate document ##
    #own_str = args.dataset

    ## random seed ##
    seed = cfg.HECO_args.seed
    numpy.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


    if cfg.HECO_args.dataset == 'mouse_human_sagittal':
        logger = setup_logger("Build logging...", cfg.HECO_args.save_path, if_train=True)

        logger.info('Configs {}\n'.format(cfg.HECO_args))
        cfg.HECO_args.target_node = 'V'
        cfg.HECO_args.nei_num = 1
        cfg.HECO_args.sample_rate = cfg.HECO.V_sample_rate
        save_path = train(cfg, logger)

        cfg.HECO_args.save_path = save_path
        cfg.HECO_args.target_node = 'S'
        cfg.HECO_args.nei_num = 1
        cfg.HECO_args.sample_rate = cfg.HECO.S_sample_rate
        save_path = train(cfg, logger)

        cfg.HECO_args.save_path = save_path
        cfg.HECO_args.nei_num = 2
        cfg.HECO_args.sample_rate = cfg.HECO.M_sample_rate
        cfg.HECO_args.target_node = 'M'
        save_path = train(cfg, logger)

        cfg.HECO_args.save_path = save_path
        cfg.HECO_args.target_node = 'H'
        cfg.HECO_args.nei_num = 2
        cfg.HECO_args.sample_rate = cfg.HECO.H_sample_rate
        save_path = train(cfg, logger)
    elif cfg.HECO_args.dataset == 'mouse_human':
        logger = setup_logger("Build logging...", cfg.HECO_args.save_path, if_train=True)

        logger.info('Configs {}\n'.format(cfg.HECO_args))
        cfg.HECO_args.target_node = 'S'
        cfg.HECO_args.nei_num = 1
        cfg.HECO_args.sample_rate = [1]

        save_path = train(cfg, logger)

        cfg.HECO_args.save_path = save_path
        cfg.HECO_args.target_node = 'V'
        cfg.HECO_args.nei_num = 1
        cfg.HECO_args.sample_rate = [2]
        save_path = train(cfg, logger)


        cfg.HECO_args.save_path = save_path
        cfg.HECO_args.nei_num = 2
        cfg.HECO_args.sample_rate = [15, 2]
        cfg.HECO_args.target_node = 'M'
        save_path = train(cfg, logger)

        cfg.HECO_args.save_path = save_path
        cfg.HECO_args.target_node = 'H'
        cfg.HECO_args.nei_num = 2
        cfg.HECO_args.sample_rate = [1, 1]
        save_path = train(cfg, logger)

    elif cfg.HECO_args.dataset == 'mouse_human_binary':
        cfg.HECO_args.nei_num = 1
        cfg.HECO_args.sample_rate = cfg.HECO.M_sample_rate
        cfg.HECO_args.if_pretrained = False
        cfg.HECO_args.target_node = 'M'
        logger = setup_logger("Build logging...", cfg.HECO_args.save_path, if_train=True)
        logger.info('Configs {}\n'.format(cfg.HECO_args))

        save_path = train(cfg, logger)
        cfg.HECO_args.save_path = save_path
        cfg.HECO_args.target_node = 'S'
        cfg.HECO_args.nei_num = 1
        cfg.HECO_args.sample_rate = cfg.HECO.S_sample_rate
        if cfg.HECO_args.if_pretrained == True:
            # args.if_pretrained = True
            # args.pretrained_model_path = "../data/mouse_human_binary/results/2022-11-30_17-18-47/"
            cfg.HECO_args.save_path = cfg.HECO_args.pretrained_model_path
        save_path = train(cfg, logger)
    elif cfg.HECO_args.dataset == 'mouse_human_three':
        cfg.HECO_args.target_node = 'S'
        cfg.HECO_args.nei_num = 1
        cfg.HECO_args.sample_rate = cfg.HECO.S_sample_rate
        if cfg.HECO_args.if_pretrained == True:
            # args.if_pretrained = True
            # args.pretrained_model_path = "../data/mouse_human_binary/results/2022-11-30_17-18-47/"
            cfg.HECO_args.save_path = cfg.HECO_args.pretrained_model_path
        logger = setup_logger("Build logging...", cfg.HECO_args.save_path, if_train=True)

        logger.info('Configs {}\n'.format(cfg.HECO_args))
        save_path = train(cfg, logger)

        cfg.HECO_args.target_node = 'G'
        cfg.HECO_args.save_path = save_path
        cfg.HECO_args.nei_num = 2
        cfg.HECO_args.sample_rate = cfg.HECO.G_sample_rate
        cfg.HECO_args.if_pretrained = False
        save_path = train(cfg, logger)

        cfg.HECO_args.target_node = 'V'
        cfg.HECO_args.save_path = save_path
        cfg.HECO_args.nei_num = 1
        cfg.HECO_args.sample_rate = cfg.HECO.V_sample_rate
        cfg.HECO_args.if_pretrained = False
        save_path = train(cfg, logger)
    else:
        logger = setup_logger("Build logging...", cfg.HECO_args.save_path, if_train=True)

        logger.info('Configs {}\n'.format(cfg.HECO_args))
        save_path = train(cfg, logger)


