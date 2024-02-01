import argparse
import sys
import time

#argv = sys.argv
#dataset = argv[1] #'mouse_human_sagittal'#
#argv[1] = dataset

def acm_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true", default=True)
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="acm")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=10000)
    
    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.05)
    parser.add_argument('--eva_wd', type=float, default=0)
    
    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0008)
    parser.add_argument('--l2_coef', type=float, default=0)
    
    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.8)
    parser.add_argument('--feat_drop', type=float, default=0.3)
    parser.add_argument('--attn_drop', type=float, default=0.5)
    parser.add_argument('--sample_rate', nargs='+', type=int, default=[7, 1])
    parser.add_argument('--lam', type=float, default=0.5)
    
    args, _ = parser.parse_known_args()
    args.type_num = [4019, 7167, 60]  # the number of every node type
    args.nei_num = 2  # the number of neighbors' types
    return args


def dblp_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="dblp")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=53)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=10000)
    
    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)
    
    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0008)
    parser.add_argument('--l2_coef', type=float, default=0)
    
    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.9)
    parser.add_argument('--feat_drop', type=float, default=0.4)
    parser.add_argument('--attn_drop', type=float, default=0.35)
    parser.add_argument('--sample_rate', nargs='+', type=int, default=[6])
    parser.add_argument('--lam', type=float, default=0.5) 

    args, _ = parser.parse_known_args()
    args.type_num = [4057, 14328, 7723, 20]  # the number of every node type
    args.nei_num = 1  # the number of neighbors' types
    return args


def aminer_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="aminer")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=10000)

    parser.add_argument('--target_node', type=str, default="p")  # S, M, H, V

    parser.add_argument('--if_pretrained', type=bool, default=False)
    parser.add_argument('--pretrained_model_path', type=str,
                        default="../data/aminer/results/")
    parser.add_argument('--save_path', type=str,
                        default="../data/aminer/results/" + time.strftime("%Y-%m-%d_%H-%M-%S") + '/')
    
    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)
    
    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--l2_coef', type=float, default=0)
    
    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--feat_drop', type=float, default=0.5)
    parser.add_argument('--attn_drop', type=float, default=0.5)
    parser.add_argument('--sample_rate', nargs='+', type=int, default=[3, 8])
    parser.add_argument('--lam', type=float, default=0.5)

    args, _ = parser.parse_known_args()
    args.type_num = [6564, 13329, 35890]  # the number of every node type
    args.nei_num = 2  # the number of neighbors' types
    return args


def freebase_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)    
    parser.add_argument('--dataset', type=str, default="freebase")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=10000)
    
    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)
    
    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2_coef', type=float, default=0)
    
    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--feat_drop', type=float, default=0.1)
    parser.add_argument('--attn_drop', type=float, default=0.3)
    parser.add_argument('--sample_rate', nargs='+', type=int, default=[1, 18, 2])
    parser.add_argument('--lam', type=float, default=0.5)
    
    args, _ = parser.parse_known_args()
    args.type_num = [3492, 2502, 33401, 4459]  # the number of every node type
    args.nei_num = 3  # the number of neighbors' types
    return args


def mouse_human_sagittal_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', type=str, default=True)
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="mouse_human_sagittal")

    parser.add_argument('--target_node', type=str, default="S") # S, M, H, V

    parser.add_argument('--if_pretrained', type=bool, default=False)
    parser.add_argument('--pretrained_model_path', type=str, default="../data/mouse_human_sagittal/results/2022-10-16_18-30-47/")
    parser.add_argument('--save_path', type=str, default="../data/mouse_human_sagittal/results/"+time.strftime("%Y-%m-%d_%H-%M-%S")+'/')

    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=53)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--nb_epochs', type=int, default=1000)

    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--l2_coef', type=float, default=0)

    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.9)
    parser.add_argument('--feat_drop', type=float, default=0.4)
    parser.add_argument('--attn_drop', type=float, default=0.35)
    parser.add_argument('--sample_rate', nargs='+', type=int, default=[6])
    parser.add_argument('--lam', type=float, default=0.5)

    args, _ = parser.parse_known_args()
    args.type_num = [21749, 4035, 6507, 3682]  # the number of every node type
    args.nei_num = 1  # the number of neighbors' types
    return args


def mouse_human_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', type=str, default=True)
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="mouse_human")

    parser.add_argument('--target_node', type=str, default="S") # S, M, H, V

    parser.add_argument('--if_pretrained', type=bool, default=False)
    parser.add_argument('--pretrained_model_path', type=str, default="../data/mouse_human_sagittal/results/2022-10-16_18-30-47/")
    parser.add_argument('--save_path', type=str, default="../data/mouse_human/results/"+time.strftime("%Y-%m-%d_%H-%M-%S")+'/')

    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=53)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--nb_epochs', type=int, default=1000)

    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--l2_coef', type=float, default=0)

    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.9)
    parser.add_argument('--feat_drop', type=float, default=0.4)
    parser.add_argument('--attn_drop', type=float, default=0.35)
    parser.add_argument('--sample_rate', nargs='+', type=int, default=[6])
    parser.add_argument('--lam', type=float, default=0.5)

    args, _ = parser.parse_known_args()
    args.type_num = [72968, 2578, 3326, 3682]  # the number of every node type
    args.nei_num = 1  # the number of neighbors' types
    return args


def mouse_human_binary_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', type=str, default=True)
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="mouse_human_binary")

    parser.add_argument('--target_node', type=str, default="S") # S, M, H, V

    parser.add_argument('--if_pretrained', type=bool, default=False)
    parser.add_argument('--pretrained_model_path', type=str, default="../data/mouse_human_binary/results/2022-10-16_18-30-47/")
    parser.add_argument('--save_path', type=str, default="../data/mouse_human_binary/results/"+time.strftime("%Y-%m-%d_%H-%M-%S")+'/')

    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=53)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--nb_epochs', type=int, default=1000)

    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--l2_coef', type=float, default=0)

    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.9)
    parser.add_argument('--feat_drop', type=float, default=0.4)
    parser.add_argument('--attn_drop', type=float, default=0.35)
    parser.add_argument('--sample_rate', nargs='+', type=int, default=[6])
    parser.add_argument('--lam', type=float, default=0.5)

    args, _ = parser.parse_known_args()
    args.type_num = [25431, 10542]  # the number of every node type
    args.nei_num = 1  # the number of neighbors' types
    return args

def mouse_human_all_binary_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', type=str, default=True)
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="mouse_human_all_binary")

    parser.add_argument('--target_node', type=str, default="S") # S, M, H, V

    parser.add_argument('--if_pretrained', type=bool, default=False)
    parser.add_argument('--pretrained_model_path', type=str, default="../data/mouse_human_all_binary/results/2022-10-16_18-30-47/")
    parser.add_argument('--save_path', type=str, default="../data/mouse_human_all_binary/results/"+time.strftime("%Y-%m-%d_%H-%M-%S")+'/')

    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=53)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--nb_epochs', type=int, default=1000)

    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--l2_coef', type=float, default=0)

    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.9)
    parser.add_argument('--feat_drop', type=float, default=0.4)
    parser.add_argument('--attn_drop', type=float, default=0.35)
    parser.add_argument('--sample_rate', nargs='+', type=int, default=[6])
    parser.add_argument('--lam', type=float, default=0.5)

    args, _ = parser.parse_known_args()
    args.type_num = [76650, 4071]  # the number of every node type
    args.nei_num = 1  # the number of neighbors' types
    return args

def set_params(dataset):
    if dataset == "acm":
        args = acm_params()
    elif dataset == "dblp":
        args = dblp_params()
    elif dataset == "aminer":
        args = aminer_params()
    elif dataset == "freebase":
        args = freebase_params()
    elif dataset == "mouse_human_binary":
        args = mouse_human_binary_params()
    elif dataset == "mouse_human_all_binary":
        args = mouse_human_all_binary_params()
    elif dataset == "mouse_human_sagittal":
        args = mouse_human_sagittal_params()
    elif dataset == "mouse_human":
        args = mouse_human_params()
    return args
