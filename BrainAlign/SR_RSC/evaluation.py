from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import logging
import warnings
warnings.filterwarnings("ignore")

def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class evaluation_metrics():
    def __init__(self, embs, labels, logger):

        self.embs = embs
        train, val, test = labels

        self.logger = logger
        
        self.trX, self.trY = self.embs[np.array(train)[:,0]], np.array(train)[:,1]
        self.valX, self.valY = self.embs[np.array(val)[:,0]], np.array(val)[:,1]
        self.tsX, self.tsY = self.embs[np.array(test)[:,0]], np.array(test)[:,1]
        self.n_label = len(set(self.tsY))
        
        self.val_acc = self.evaluate_cluster()
        
        
    def evaluation_lp(self, node1, node2, label):

        X1, X2 = [], []
        cnt = 0
        error = 0
        prob = []
        preds = []

        meanvec = np.mean(self.embs, 0)
        for i in range(len(node1)):
            n1 = int(node1[i])
            n2 = int(node2[i])
            X1 = self.embs[n1]
            X2 = self.embs[n2]

            if X1.sum() == 0:
                cnt+= 1
                X1 = meanvec
            if X2.sum() == 0:
                cnt+= 1
                X2 = meanvec
            r = X1.dot(X2)
            prob.append(r)
            if r >= 0.5:
                r = 1
            else:
                r = 0
            preds.append(r)
            if r != label[i]:
                error += 1

        auc = metrics.roc_auc_score(label, prob)
        precision, recall, thresholds = metrics.precision_recall_curve(label, prob)
        pr = metrics.auc(recall, precision)
        ap = metrics.average_precision_score(label, prob, average=None)
        acc = metrics.accuracy_score(label, preds)
        f1_micro = metrics.f1_score(label, preds, average='micro')
        f1_macro = metrics.f1_score(label, preds, average='macro')
        self.logger.info('AUC: %.5f, AP: %.5f, PR: %.5f, ACC: %.5f, F1_micro: %.5f, F1_macro: %.5f'%(auc, ap, pr, acc, f1_micro, f1_macro))

    def evalutation(self):
        
        nmis, adjscores, puritys, fis, fas = 0,0,0,0,0
#         for rs in [0,123,432,6543,8478643]:
        for rs in [0]:
            kmeans = KMeans(n_clusters=self.n_label, random_state=rs).fit(self.tsX)
            preds = kmeans.predict(self.tsX)
            nmi = metrics.normalized_mutual_info_score(labels_true=self.tsY, labels_pred=np.array(preds))
            adjscore = metrics.adjusted_rand_score(self.tsY, np.array(preds))
            purity = purity_score(self.tsY, np.array(preds))
            nmis += nmi
            adjscores += adjscore
            puritys+=purity

            lr = LogisticRegression(max_iter=500, random_state=rs, solver='sag')
            lr.fit(self.trX, self.trY)
            Y_pred = lr.predict(self.tsX)
            f1_micro = metrics.f1_score(self.tsY, Y_pred, average='micro')
            f1_macro = metrics.f1_score(self.tsY, Y_pred, average='macro')
            fis+=f1_micro
            fas+=f1_macro
        self.logger.info('NMI=%.5f, ARI: %.5f, f1_micro=%.5f, f1_macro=%.5f' % (nmis, adjscores, fis, fas))


    def evaluate_cluster(self):

        kmeans = KMeans(n_clusters=self.n_label, random_state=0).fit(self.valX)
        preds = kmeans.predict(self.trX)
        nmi = metrics.normalized_mutual_info_score( labels_true=self.trY, labels_pred=np.array(preds))
        return nmi
    
    def evaluate_clf(self):
        r"""Evaluates latent space quality via a logistic regression downstream task."""
        clf = LogisticRegression(max_iter=500, random_state=0, solver='lbfgs').fit(self.trX, self.trY)
        val_acc = clf.score(self.valX, self.valY)
        return val_acc
    
