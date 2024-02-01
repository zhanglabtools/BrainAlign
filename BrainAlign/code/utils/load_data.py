import numpy as np
import scipy.sparse as sp
import torch as th
from sklearn.preprocessing import OneHotEncoder


def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    #sparse_mx = sparse_mx.tocoo().astype(np.float16)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    #indices = th.from_numpy(
    #    np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int16))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)


def load_acm(ratio, type_num):
    # The order of node types: 0 p 1 a 2 s
    path = "../data/acm/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    nei_s = np.load(path + "nei_s.npy", allow_pickle=True)
    feat_p = sp.load_npz(path + "p_feat.npz")
    feat_a = sp.eye(type_num[1])
    feat_s = sp.eye(type_num[2])
    pap = sp.load_npz(path + "pap.npz")
    psp = sp.load_npz(path + "psp.npz")
    pos = sp.load_npz(path + "pos.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)
    nei_a = [th.LongTensor(i) for i in nei_a]
    nei_s = [th.LongTensor(i) for i in nei_s]
    feat_p = th.FloatTensor(preprocess_features(feat_p))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_s = th.FloatTensor(preprocess_features(feat_s))
    pap = sparse_mx_to_torch_sparse_tensor(normalize_adj(pap))
    psp = sparse_mx_to_torch_sparse_tensor(normalize_adj(psp))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    return [nei_a, nei_s], [feat_p, feat_a, feat_s], [pap, psp], pos, label, train, val, test



def load_mouse_human(ratio, type_num, path = "../data/mouse_human/data/", target_node = 'S'):
    # The order of node types: 0 s 1 m 2 h 3 v

    label = np.load(path + "labels.npy").astype('int32')
    #label = encode_onehot(label)

    feat_s = sp.load_npz(path + "s_feat.npz").astype("float16")
    feat_m = sp.load_npz(path + "m_feat.npz").astype("float16")
    feat_h = sp.load_npz(path + "h_feat.npz").astype("float16")
    feat_v = sp.load_npz(path + "v_feat.npz").astype("float16")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]
    label = th.FloatTensor(label)
    feat_s = th.FloatTensor(preprocess_features(feat_s))
    feat_m = th.FloatTensor(preprocess_features(feat_m))
    feat_h = th.FloatTensor(preprocess_features(feat_h))
    feat_v = th.FloatTensor(preprocess_features(feat_v))
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]

    if target_node == 'S':
        nei_s = np.load(path + "s_nei_m.npy", allow_pickle=True)
        nei_s = [th.LongTensor(i) for i in nei_s]
        nei_list = [nei_s]

        sms = sp.load_npz(path + "sms.npz").astype("float16")
        smhvhms = sp.load_npz(path + "smhvhms.npz").astype("float16")
        #smhvvhms = sp.load_npz(path + "smhvvhms.npz").astype("float16")
        sms = sparse_mx_to_torch_sparse_tensor(normalize_adj(sms))
        smhvhms = sparse_mx_to_torch_sparse_tensor(normalize_adj(smhvhms))
        #smhvvhms = sparse_mx_to_torch_sparse_tensor(normalize_adj(smhvvhms))
        #meta_path_list = [sms, smhvhms, smhvvhms]
        meta_path_list = [sms, smhvhms]
        feat_list = [feat_s, feat_m, feat_h, feat_v]

        pos = sp.load_npz(path + "s_pos.npz").astype("float16")
        pos = sparse_mx_to_torch_sparse_tensor(pos)
    elif target_node == 'M':
        nei_m_s = np.load(path + "m_nei_s.npy", allow_pickle=True)
        nei_m_h = np.load(path + "m_nei_h.npy", allow_pickle=True)
        nei_m_s = [th.LongTensor(i) for i in nei_m_s]
        nei_m_h = [th.LongTensor(i) for i in nei_m_h]
        nei_list = [nei_m_s, nei_m_h]

        msm = sp.load_npz(path + "msm.npz").astype("float16")
        mssm = sp.load_npz(path + "mssm.npz").astype("float16")
        mhvhm = sp.load_npz(path + "mhvhm.npz").astype("float16")
        mhvvhm = sp.load_npz(path + "mhvvhm.npz").astype("float16")
        msm = sparse_mx_to_torch_sparse_tensor(normalize_adj(msm))
        mssm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mssm))
        mhvhm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mhvhm))
        mhvvhm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mhvvhm))
        meta_path_list = [msm, mssm, mhvhm, mhvvhm]

        feat_list = [feat_m, feat_s, feat_h, feat_v]

        pos = sp.load_npz(path + "m_pos.npz").astype("float16")
        pos = sparse_mx_to_torch_sparse_tensor(pos)
    elif target_node == 'H':
        nei_h_m = np.load(path + "m_nei_m.npy", allow_pickle=True)
        nei_h_v = np.load(path + "m_nei_v.npy", allow_pickle=True)
        nei_h_m = [th.LongTensor(i) for i in nei_h_m]
        nei_h_v = [th.LongTensor(i) for i in nei_h_v]
        nei_list = [nei_h_m, nei_h_v]

        hvh = sp.load_npz(path + "hvh.npz").astype("float16")
        hvvh = sp.load_npz(path + "hvvh.npz").astype("float16")
        hmsmh = sp.load_npz(path + "hmsmh.npz").astype("float16")
        hmssmh = sp.load_npz(path + "hmssmh.npz").astype("float16")
        hvh = sparse_mx_to_torch_sparse_tensor(normalize_adj(hvh))
        hvvh = sparse_mx_to_torch_sparse_tensor(normalize_adj(hvvh))
        hmsmh = sparse_mx_to_torch_sparse_tensor(normalize_adj(hmsmh))
        hmssmh = sparse_mx_to_torch_sparse_tensor(normalize_adj(hmssmh))
        meta_path_list = [hvh, hvvh, hmsmh, hmssmh]

        feat_list = [feat_h, feat_m, feat_v, feat_s]

        pos = sp.load_npz(path + "h_pos.npz").astype("float16")
        pos = sparse_mx_to_torch_sparse_tensor(pos)
    elif target_node == 'V':
        nei_v = np.load(path + "v_nei_h.npy", allow_pickle=True)
        nei_v = [th.LongTensor(i) for i in nei_v]
        nei_list = [nei_v]

        vhv = sp.load_npz(path + "vhv.npz").astype("float16")
        vhmsmhv = sp.load_npz(path + "vhmsmhv.npz").astype("float16")
        vhmssmhv = sp.load_npz(path + "vhmssmhv.npz").astype("float16")
        vhv = sparse_mx_to_torch_sparse_tensor(normalize_adj(vhv))
        vhmsmhv = sparse_mx_to_torch_sparse_tensor(normalize_adj(vhmsmhv))
        vhmssmhv = sparse_mx_to_torch_sparse_tensor(normalize_adj(vhmssmhv))
        meta_path_list = [vhv, vhmsmhv, vhmssmhv]

        feat_list = [feat_v, feat_h, feat_m, feat_s]

        pos = sp.load_npz(path + "v_pos.npz").astype("float16")
        pos = sparse_mx_to_torch_sparse_tensor(pos)

    return nei_list, feat_list, meta_path_list, pos, label, train, val, test


def load_mouse_human_sagittal(ratio, type_num, path = "../data/mouse_human_sagittal/data/", target_node = 'S'):
    # The order of node types: 0 s 1 m 2 h 3 v

    label = np.load(path + "labels.npy").astype('int32')
    #label = encode_onehot(label)

    feat_s = sp.load_npz(path + "s_feat.npz").astype("float16")
    feat_m = sp.load_npz(path + "m_feat.npz").astype("float16")
    feat_h = sp.load_npz(path + "h_feat.npz").astype("float16")
    feat_v = sp.load_npz(path + "v_feat.npz").astype("float16")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]
    label = th.FloatTensor(label)
    feat_s = th.FloatTensor(preprocess_features(feat_s))
    feat_m = th.FloatTensor(preprocess_features(feat_m))
    feat_h = th.FloatTensor(preprocess_features(feat_h))
    feat_v = th.FloatTensor(preprocess_features(feat_v))
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]

    if target_node == 'S':
        nei_s = np.load(path + "s_nei_m.npy", allow_pickle=True)
        nei_s = [th.LongTensor(i) for i in nei_s]
        nei_list = [nei_s]

        sms = sp.load_npz(path + "sms.npz").astype("float16")
        smhvhms = sp.load_npz(path + "smhvhms.npz").astype("float16")
        smhvvhms = sp.load_npz(path + "smhvvhms.npz").astype("float16")
        sms = sparse_mx_to_torch_sparse_tensor(normalize_adj(sms))
        smhvhms = sparse_mx_to_torch_sparse_tensor(normalize_adj(smhvhms))
        smhvvhms = sparse_mx_to_torch_sparse_tensor(normalize_adj(smhvvhms))
        meta_path_list = [sms, smhvhms, smhvvhms]
        feat_list = [feat_s, feat_m, feat_h, feat_v]

        pos = sp.load_npz(path + "s_pos.npz").astype("float16")
        pos = sparse_mx_to_torch_sparse_tensor(pos)
    elif target_node == 'M':
        nei_m_s = np.load(path + "m_nei_s.npy", allow_pickle=True)
        nei_m_h = np.load(path + "m_nei_h.npy", allow_pickle=True)
        nei_m_s = [th.LongTensor(i) for i in nei_m_s]
        nei_m_h = [th.LongTensor(i) for i in nei_m_h]
        nei_list = [nei_m_s, nei_m_h]

        msm = sp.load_npz(path + "msm.npz").astype("float16")
        mssm = sp.load_npz(path + "mssm.npz").astype("float16")
        mhvhm = sp.load_npz(path + "mhvhm.npz").astype("float16")
        mhvvhm = sp.load_npz(path + "mhvvhm.npz").astype("float16")
        msm = sparse_mx_to_torch_sparse_tensor(normalize_adj(msm))
        mssm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mssm))
        mhvhm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mhvhm))
        mhvvhm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mhvvhm))
        meta_path_list = [msm, mssm, mhvhm, mhvvhm]

        feat_list = [feat_m, feat_s, feat_h, feat_v]

        pos = sp.load_npz(path + "m_pos.npz").astype("float16")
        pos = sparse_mx_to_torch_sparse_tensor(pos)
    elif target_node == 'H':
        nei_h_m = np.load(path + "h_nei_m.npy", allow_pickle=True)
        nei_h_v = np.load(path + "h_nei_v.npy", allow_pickle=True)
        nei_h_m = [th.LongTensor(i) for i in nei_h_m]
        nei_h_v = [th.LongTensor(i) for i in nei_h_v]
        nei_list = [nei_h_m, nei_h_v]

        hvh = sp.load_npz(path + "hvh.npz").astype("float16")
        hvvh = sp.load_npz(path + "hvvh.npz").astype("float16")
        hmsmh = sp.load_npz(path + "hmsmh.npz").astype("float16")
        hmssmh = sp.load_npz(path + "hmssmh.npz").astype("float16")
        hvh = sparse_mx_to_torch_sparse_tensor(normalize_adj(hvh))
        hvvh = sparse_mx_to_torch_sparse_tensor(normalize_adj(hvvh))
        hmsmh = sparse_mx_to_torch_sparse_tensor(normalize_adj(hmsmh))
        hmssmh = sparse_mx_to_torch_sparse_tensor(normalize_adj(hmssmh))
        meta_path_list = [hvh, hvvh, hmsmh, hmssmh]

        feat_list = [feat_h, feat_m, feat_v, feat_s]

        pos = sp.load_npz(path + "h_pos.npz").astype("float16")
        pos = sparse_mx_to_torch_sparse_tensor(pos)
    elif target_node == 'V':
        nei_v = np.load(path + "v_nei_h.npy", allow_pickle=True)
        nei_v = [th.LongTensor(i) for i in nei_v]
        nei_list = [nei_v]

        vhv = sp.load_npz(path + "vhv.npz").astype("float16")
        vhmsmhv = sp.load_npz(path + "vhmsmhv.npz").astype("float16")
        vhmssmhv = sp.load_npz(path + "vhmssmhv.npz").astype("float16")
        vhv = sparse_mx_to_torch_sparse_tensor(normalize_adj(vhv))
        vhmsmhv = sparse_mx_to_torch_sparse_tensor(normalize_adj(vhmsmhv))
        vhmssmhv = sparse_mx_to_torch_sparse_tensor(normalize_adj(vhmssmhv))
        meta_path_list = [vhv, vhmsmhv, vhmssmhv]

        feat_list = [feat_v, feat_h, feat_m, feat_s]

        pos = sp.load_npz(path + "v_pos.npz").astype("float16")
        pos = sparse_mx_to_torch_sparse_tensor(pos)


    return nei_list, feat_list, meta_path_list, pos, label, train, val, test
    #return [nei_s], [feat_s, feat_m, feat_h, feat_v], [smhvvhms], pos, label, train, val, test



def load_mouse_human_binary(ratio, type_num, path = "../data/mouse_human_binary/data/", target_node = 'S'):
    # The order of node types: 0 s 1 m 2 h 3 v
    #path = "../data/mouse_human_binary/data/"
    label = np.load(path + "labels.npy").astype('int32')
    #label = encode_onehot(label)

    feat_s = sp.load_npz(path + "s_feat.npz").astype("float16")
    feat_m = sp.load_npz(path + "m_feat.npz").astype("float16")

    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]
    label = th.FloatTensor(label)
    feat_s = th.FloatTensor(preprocess_features(feat_s))
    feat_m = th.FloatTensor(preprocess_features(feat_m))

    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]

    if target_node == 'S':
        nei_s = np.load(path + "s_nei_m.npy", allow_pickle=True)
        nei_s = [th.LongTensor(i) for i in nei_s]
        nei_list = [nei_s]

        sms = sp.load_npz(path + "sms.npz").astype("float16")
        smms = sp.load_npz(path + "smms.npz").astype("float16")
        sms = sparse_mx_to_torch_sparse_tensor(normalize_adj(sms))
        smms = sparse_mx_to_torch_sparse_tensor(normalize_adj(smms))

        meta_path_list = [sms, smms]
        feat_list = [feat_s, feat_m]

        pos = sp.load_npz(path + "s_pos.npz").astype("float16")
        pos = sparse_mx_to_torch_sparse_tensor(pos)
    elif target_node == 'M':
        nei_m_s = np.load(path + "m_nei_s.npy", allow_pickle=True)

        nei_m_s = [th.LongTensor(i) for i in nei_m_s]
        nei_list = [nei_m_s]

        msm = sp.load_npz(path + "msm.npz").astype("float16")
        mssm = sp.load_npz(path + "mssm.npz").astype("float16")

        msm = sparse_mx_to_torch_sparse_tensor(normalize_adj(msm))
        mssm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mssm))

        meta_path_list = [msm, mssm]

        feat_list = [feat_m, feat_s]

        pos = sp.load_npz(path + "m_pos.npz").astype("float16")
        pos = sparse_mx_to_torch_sparse_tensor(pos)

    return nei_list, feat_list, meta_path_list, pos, label, train, val, test



def load_mouse_human_three(ratio, type_num, path = "../data/mouse_human_sagittal/data/", target_node = 'S'):
    # The order of node types: 0 s 1 m 2 h 3 v

    label = np.load(path + "labels.npy").astype('int32')
    #label = encode_onehot(label)

    feat_s = sp.load_npz(path + "s_feat.npz").astype("float16")
    feat_g = sp.load_npz(path + "g_feat.npz").astype("float16")
    feat_v = sp.load_npz(path + "v_feat.npz").astype("float16")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]
    label = th.FloatTensor(label)
    feat_s = th.FloatTensor(preprocess_features(feat_s))
    feat_g = th.FloatTensor(preprocess_features(feat_g))
    feat_v = th.FloatTensor(preprocess_features(feat_v))
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]

    if target_node == 'S':
        nei_s = np.load(path + "s_nei_g.npy", allow_pickle=True)
        nei_s = [th.LongTensor(i) for i in nei_s]
        nei_list = [nei_s]

        sgs = sp.load_npz(path + "sgs.npz").astype("float16")
        sgvgs = sp.load_npz(path + "sgvgs.npz").astype("float16")
        sgvvgs = sp.load_npz(path + "sgvvgs.npz").astype("float16")
        sgs = sparse_mx_to_torch_sparse_tensor(normalize_adj(sgs))
        sgvgs = sparse_mx_to_torch_sparse_tensor(normalize_adj(sgvgs))
        sgvvgs = sparse_mx_to_torch_sparse_tensor(normalize_adj(sgvvgs))
        meta_path_list = [sgs, sgvgs, sgvvgs]
        feat_list = [feat_s, feat_g, feat_v]

        pos = sp.load_npz(path + "s_pos.npz").astype("float16")
        pos = sparse_mx_to_torch_sparse_tensor(pos)
    elif target_node == 'G':
        nei_g_s = np.load(path + "g_nei_s.npy", allow_pickle=True)
        nei_g_v = np.load(path + "g_nei_v.npy", allow_pickle=True)
        nei_g_s = [th.LongTensor(i) for i in nei_g_s]
        nei_g_v = [th.LongTensor(i) for i in nei_g_v]
        nei_list = [nei_g_s, nei_g_v]

        gsg = sp.load_npz(path + "gsg.npz").astype("float16")
        gssg = sp.load_npz(path + "gssg.npz").astype("float16")
        gvg = sp.load_npz(path + "gvg.npz").astype("float16")
        gvvg = sp.load_npz(path + "gvvg.npz").astype("float16")
        gsg = sparse_mx_to_torch_sparse_tensor(normalize_adj(gsg))
        gssg = sparse_mx_to_torch_sparse_tensor(normalize_adj(gssg))
        gvg = sparse_mx_to_torch_sparse_tensor(normalize_adj(gvg))
        gvvg = sparse_mx_to_torch_sparse_tensor(normalize_adj(gvvg))
        meta_path_list = [gsg, gssg, gvg, gvvg]

        feat_list = [feat_g, feat_s, feat_v]

        pos = sp.load_npz(path + "g_pos.npz").astype("float16")
        pos = sparse_mx_to_torch_sparse_tensor(pos)

    elif target_node == 'V':
        nei_v = np.load(path + "v_nei_g.npy", allow_pickle=True)
        nei_v = [th.LongTensor(i) for i in nei_v]
        nei_list = [nei_v]

        vgv = sp.load_npz(path + "vgv.npz").astype("float16")
        vggv = sp.load_npz(path + "vggv.npz").astype("float16")
        meta_path_list = [vgv, vggv]

        feat_list = [feat_v, feat_g, feat_s]

        pos = sp.load_npz(path + "v_pos.npz").astype("float16")
        pos = sparse_mx_to_torch_sparse_tensor(pos)


    return nei_list, feat_list, meta_path_list, pos, label, train, val, test


def load_dblp(ratio, type_num):
    # The order of node types: 0 a 1 p 2 c 3 t
    path = "../data/dblp/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    nei_p = np.load(path + "nei_p.npy", allow_pickle=True)
    feat_a = sp.load_npz(path + "a_feat.npz").astype("float32")
    feat_p = sp.eye(type_num[1])
    apa = sp.load_npz(path + "apa.npz")
    apcpa = sp.load_npz(path + "apcpa.npz")
    aptpa = sp.load_npz(path + "aptpa.npz")
    pos = sp.load_npz(path + "pos.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]
    
    label = th.FloatTensor(label)
    nei_p = [th.LongTensor(i) for i in nei_p]
    feat_p = th.FloatTensor(preprocess_features(feat_p))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    apa = sparse_mx_to_torch_sparse_tensor(normalize_adj(apa))
    apcpa = sparse_mx_to_torch_sparse_tensor(normalize_adj(apcpa))
    aptpa = sparse_mx_to_torch_sparse_tensor(normalize_adj(aptpa))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    return [nei_p], [feat_a, feat_p], [apa, apcpa, aptpa], pos, label, train, val, test


def load_aminer(ratio, type_num):
    # The order of node types: 0 p 1 a 2 r
    path = "../data/aminer/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    nei_r = np.load(path + "nei_r.npy", allow_pickle=True)
    # Because none of P, A or R has features, we assign one-hot encodings to all of them.
    feat_p = sp.eye(type_num[0])
    feat_a = sp.eye(type_num[1])
    feat_r = sp.eye(type_num[2])
    pap = sp.load_npz(path + "pap.npz")
    prp = sp.load_npz(path + "prp.npz")
    pos = sp.load_npz(path + "pos.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)
    nei_a = [th.LongTensor(i) for i in nei_a]
    nei_r = [th.LongTensor(i) for i in nei_r]
    feat_p = th.FloatTensor(preprocess_features(feat_p))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_r = th.FloatTensor(preprocess_features(feat_r))
    pap = sparse_mx_to_torch_sparse_tensor(normalize_adj(pap))
    prp = sparse_mx_to_torch_sparse_tensor(normalize_adj(prp))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    return [nei_a, nei_r], [feat_p, feat_a, feat_r], [pap, prp], pos, label, train, val, test


def load_freebase(ratio, type_num):
    # The order of node types: 0 m 1 d 2 a 3 w
    path = "../data/freebase/"
    #label = np.load(path + "labels.npy").astype('int32')
    #label = encode_onehot(label)
    nei_d = np.load(path + "nei_d.npy", allow_pickle=True)
    nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    nei_w = np.load(path + "nei_w.npy", allow_pickle=True)
    feat_m = sp.eye(type_num[0])
    feat_d = sp.eye(type_num[1])
    feat_a = sp.eye(type_num[2])
    feat_w = sp.eye(type_num[3])
    # Because none of M, D, A or W has features, we assign one-hot encodings to all of them.
    mam = sp.load_npz(path + "mam.npz")
    mdm = sp.load_npz(path + "mdm.npz")
    mwm = sp.load_npz(path + "mwm.npz")
    pos = sp.load_npz(path + "pos.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    #label = th.FloatTensor(label)
    nei_d = [th.LongTensor(i) for i in nei_d]
    nei_a = [th.LongTensor(i) for i in nei_a]
    nei_w = [th.LongTensor(i) for i in nei_w]
    feat_m = th.FloatTensor(preprocess_features(feat_m))
    feat_d = th.FloatTensor(preprocess_features(feat_d))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_w = th.FloatTensor(preprocess_features(feat_w))
    mam = sparse_mx_to_torch_sparse_tensor(normalize_adj(mam))
    mdm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mdm))
    mwm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mwm))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    return [nei_d, nei_a, nei_w], [feat_m, feat_d, feat_a, feat_w], [mdm, mam, mwm], pos, label, train, val, test


def load_data(cfg, dataset, ratio, type_num, target_node = 'S'):
    if dataset == "acm":
        data = load_acm(ratio, type_num)
    elif dataset == "dblp":
        data = load_dblp(ratio, type_num)
    elif dataset == "aminer":
        data = load_aminer(ratio, type_num)
    elif dataset == "freebase":
        data = load_freebase(ratio, type_num)
    elif dataset == "mouse_human_binary":
        path = cfg.HECO_args.data_path
        data = load_mouse_human_binary(ratio, type_num, path=path, target_node = target_node)
    elif dataset == "mouse_human_three":
        path = cfg.HECO_args.data_path
        data = load_mouse_human_three(ratio, type_num, path=path, target_node = target_node)
    elif dataset == "mouse_human_all_binary":
        path = cfg.HECO_args.data_path
        data = load_mouse_human_binary(ratio, type_num, path=path, target_node = target_node)
    elif dataset == "mouse_human_sagittal":
        path = cfg.HECO_args.data_path
        data = load_mouse_human_sagittal(ratio, type_num,  path=path, target_node = target_node)
    elif dataset == "mouse_human":
        path = cfg.HECO_args.data_path
        data = load_mouse_human(ratio, type_num,  path=path, target_node = target_node)
    return data
