from sklearn.preprocessing import normalize
import scipy.sparse as sp
import torch
import torch.nn as nn
import numpy as np

def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    labels = np.eye(nb_classes)[targets]
    return torch.LongTensor(labels)


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1.0).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_to_tuple(mx):
#     mx = normalize_adj(mx)
    if not sp.isspmatrix_coo(mx):
        mx = mx.tocoo()
    coords = np.vstack((mx.row, mx.col))
    values = mx.data
    shape = mx.shape
    return coords, values, shape


def preprocess_features(features, norm=True):
    """Row-normalize feature matrix and convert to tuple representation"""
    if sp.issparse(features):
        features = features.toarray()
    if norm:
        features[features>0] = 1
#         rowsum = np.array(features.sum(1))
#         r_inv = np.power(rowsum, -1.0).flatten()
#         r_inv[np.isinf(r_inv)] = 0.
#         r_mat_inv = sp.diags(r_inv)
#         features = r_mat_inv.dot(features)
    return torch.FloatTensor(features)


def normalize_mx(mx, diagonal=True):
    if diagonal:
        size = mx.shape[0]
        return normalize(mx+sp.eye(size), norm='l1', axis=1)
    else:
        return normalize(mx, norm='l1', axis=1)



        
