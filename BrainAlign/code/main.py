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
import sys
import argparse

def train(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    logger = setup_logger("Build logging...", args.save_path, if_train=True)

    logger.info('Configs {}\n'.format(args))

    nei_index, feats, mps, pos, label, idx_train, idx_val, idx_test = \
        load_data(args.dataset, args.ratio, args.type_num, args.target_node)
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
        if not os.path.exists(args.save_path+"/embeds/"+"/"):
            os.makedirs(args.save_path+"/embeds/"+"/")
        f = open(args.save_path+"/embeds/"+"/"+"node_"+args.target_node+".pkl", "wb")
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


if __name__ == '__main__':
    #save_path = '../data/mouse_human_sagittal/results/'
    dataset = sys.argv[1]
    args = set_params(dataset)
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.gpu))
        torch.cuda.set_device(args.gpu)
    else:
        device = torch.device("cpu")

    ## name of intermediate document ##
    #own_str = args.dataset

    ## random seed ##
    seed = args.seed
    numpy.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


    if args.dataset == 'mouse_human_sagittal':

        args.target_node = 'V'
        args.nei_num = 1
        args.sample_rate = [2]
        save_path = train(args)

        args.save_path = save_path
        args.target_node = 'S'
        args.nei_num = 1
        args.sample_rate = [1]
        save_path = train(args)

        args.save_path = save_path
        args.nei_num = 2
        args.sample_rate = [5, 2]
        args.target_node = 'M'
        save_path = train(args)

        args.save_path = save_path
        args.target_node = 'H'
        args.nei_num = 2
        args.sample_rate = [1, 1]
        save_path = train(args)
    elif args.dataset == 'mouse_human':

        args.target_node = 'S'
        args.nei_num = 1
        args.sample_rate = [1]
        save_path = train(args)

        args.save_path = save_path
        args.target_node = 'V'
        args.nei_num = 1
        args.sample_rate = [2]
        save_path = train(args)


        args.save_path = save_path
        args.nei_num = 2
        args.sample_rate = [15, 2]
        args.target_node = 'M'
        save_path = train(args)

        args.save_path = save_path
        args.target_node = 'H'
        args.nei_num = 2
        args.sample_rate = [1, 1]
        save_path = train(args)

    elif args.dataset == 'mouse_human_binary':
        args.target_node = 'S'
        args.nei_num = 1
        args.sample_rate = [1]
        #args.if_pretrained = True
        #args.pretrained_model_path = "../data/mouse_human_binary/results/2022-11-30_17-18-47/"
        #args.save_path = args.pretrained_model_path
        save_path = train(args)

        args.save_path = save_path
        args.nei_num = 1
        args.sample_rate = [3]
        args.if_pretrained = False
        args.target_node = 'M'
        save_path = train(args)
    else:
        save_path = train(args)

