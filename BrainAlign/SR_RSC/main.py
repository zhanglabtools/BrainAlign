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


def parse_args():
    # input arguments
    parser = argparse.ArgumentParser(description='BiHIN')
    parser.add_argument('--gpu_num', nargs='?', default='0')
    parser.add_argument('--model', nargs='?', default='SubHIN')
    parser.add_argument('--dataset', nargs='?', default='dblp')
    parser.add_argument('--save_path', nargs='?', default='./results')

    parser.add_argument('--nb_epochs', type=int, default=10000)
    parser.add_argument('--lr', type = float, default = 0.001)
    parser.add_argument('--patience', type=int, default=50)

#     parser.add_argument('--att_hid_units', type=int, default=64)
    parser.add_argument('--hid_units', type=int, default=256)# 128 best for dblp and yelp, larger datasets
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
    parser.add_argument('--isLP', action='store_true', default=False)# link prediction
    parser.add_argument('--isSemi', action='store_true', default=False)# semi-supervised learning

    return parser.parse_known_args()

def printConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))
    print(args_names)
    print(args_vals)

def main():
    args, unknown = parse_args()
#     printConfig(args)
    if args.model == 'SubHIN':
        from models import SubHIN
        embedder = SubHIN(args)
    start = time.time()
    embedder.training()
    print('time (s):%.2f'%(time.time()-start))


if __name__ == '__main__':
    main()
