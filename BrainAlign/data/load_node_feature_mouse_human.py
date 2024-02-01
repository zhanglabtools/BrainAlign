import numpy as np
import scipy.sparse as sp
import pickle

'''
   The 1st kind of method to get initial embeddings: use original sample/voxel expression data.
'''

def extract_expression_embedding():
    return 0


'''
    The 2nd kind of method to get initial embeddings: use embeddings output by CAME. 
'''
def extract_came_embedding():

    return 0



def init_embedding(method='CAME'):
    if method == 'CAME':
        extract_came_embedding()
    elif method == 'Expression':
        extract_expression_embedding()



if __name__ == '__main__':
    path_datapiar = '../../../CAME/brain_human_mouse/(\'Baron_mouse\', \'Baron_human\')-(06-19 16.19.17)/datapair_init.pickle'
    path_datapiar_file = open(path_datapiar, 'rb')
    datapair = pickle.load(path_datapiar_file)
    print(datapair)
    print(datapair['features'][0].shape)
    print(datapair['features'][1].shape)

    print(datapair['varnames_feat'])



    '''
    nei = np.load('./dblp/nei_p.npy', allow_pickle=True)
    print(nei)
    print(nei.shape)
    print(nei[0].shape)
    for arr in nei:
        print(arr.shape)
    '''

    '''
    p_feat = sp.load_npz('./dblp/p_feat.npz')
    print(p_feat.shape)
    a_feat = sp.load_npz('./dblp/a_feat.npz')
    print(a_feat.shape)
    t_feat = np.load('./dblp/t_feat.npz')
    print(t_feat.shape)
    '''





