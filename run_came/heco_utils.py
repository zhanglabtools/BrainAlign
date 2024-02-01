import numpy as np


def threshold_array(X):
    '''
    input: array row: sample, column: gene vector
    output: a binary matrix
    For each value in (i, j), the binary value = if(M_ij > avg(column_j))
    '''
    return X > np.mean(X, axis=0)

def threshold_quantile(X, quantile_gene=0.9, quantile_sample=0.95):
    '''
    input: array row: sample, column: gene vector
    output: a binary matrix
    For each value in (i, j), the binary value = if(M_ij > avg(column_j))
    '''
    keep_mat_gene = X > np.quantile(X, quantile_gene, axis=0)
    keep_mat_sample = (X.T > np.quantile(X.T, quantile_sample, axis=0)).T
    keep_mat = keep_mat_sample + keep_mat_gene
    return X * keep_mat


def threshold_top(X, percent=1):
    '''
    input: array row: sample, column: gene vector
    output: a binary matrix
    For each value in (i, j), the binary value = if(M_ij > avg(column_j))
    '''
    #topk = int(round(X.shape[0] * percent))
    topk = percent
    #print(topk)
    #topk_pos = X.shape[0] - topk
    X_sort = np.sort(X, axis=0)
    return X >= X_sort[-topk, :]


def threshold_array_nonzero(X):
    '''
    input: array row: sample, column: gene vector
    output: a binary matrix
    For each value in (i, j), the binary value = if(M_ij > avg(column_j))
    '''
    return X > 0



if __name__ == '__main__':
    X = np.array([[1,2,3],[2,3,4], [2,3,4], [4,5,2], [7,26,10]])
    print(X)
    print(threshold_top(X, percent=0.4))
    #print(threshold_array(X))