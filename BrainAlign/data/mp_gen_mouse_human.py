import numpy as np
import scipy.sparse as sp
import pickle
####################################################
# This tool is to generate meta-path based adjacency
# matrix given original links.
####################################################


path_datapiar = '../../../CAME/brain_mouse_human_no_threshold/datapair_init.pickle'
path_datapiar_file = open(path_datapiar, 'rb')
datapair = pickle.load(path_datapiar_file)
print(datapair)
print(len(datapair['varnames_node'][0]))
np.save('./mouse_human/mouse_gene_names.npy', datapair['varnames_node'][0])
print(len(datapair['varnames_node'][1]))
np.save('./mouse_human/human_gene_names.npy', datapair['varnames_node'][1])



S = 72968
M = 2578
H = 3326
V = 3682

sm_ = datapair['ov_adjs'][0].toarray()
print(sm_)
print('sm_', sm_.shape)

vh_ = datapair['ov_adjs'][1].toarray()
print('vh_', vh_.shape)
mm_ = datapair['vv_adj'].toarray()[0:M, 0:M]
print('mm_', mm_.shape)
print('mm_ sum', np.sum(mm_))
hh_ = datapair['vv_adj'].toarray()[M:, M:]
print('hh_', hh_.shape)
print('hh_ sum', np.sum(hh_)) # == 0
mh_ = datapair['vv_adj'].toarray()[0:M, M:]
print('mh_', mh_.shape)
#ss_ = datapair['oo_adjs'].toarray()[0:S, 0:S]
#print('ss_', ss_.shape)
#print('ss_ sum', np.sum(ss_)) # == 0
vv_ = datapair['oo_adjs'].toarray()[S:, S:]
print('vv_', vv_.shape)
print(np.sum(vv_))
print('vv_ sum', np.sum(vv_))
sv_ = datapair['oo_adjs'].toarray()[0:S, S:]
print('sv_', sv_.shape)

'''
sms = np.matmul(sm_, sm_.T) # > 0
print(sms)
sms = sp.coo_matrix(sms)
sp.save_npz("./mouse_human/sms.npz", sms)


smh = np.matmul(sm_, mh_) #> 0
smhv = np.matmul(smh, vh_.T) #> 0
smhvhms = np.matmul(smhv, smhv.T) #> 0
print(smhvhms)
smhvhms = sp.coo_matrix(smhvhms)
sp.save_npz("./mouse_human/smhvhms.npz", smhvhms)

smh = np.matmul(sm_, mh_) #> 0
smhv = np.matmul(smh, vh_.T) #> 0
smhvv = np.matmul(smhv, vv_) #> 0
smhvvhms = np.matmul(smhv, smhvv.T) #> 0
print(smhvvhms)
smhvvhms = sp.coo_matrix(smhvvhms)
sp.save_npz("./mouse_human/smhvvhms.npz", smhvvhms)


'''



'''
sms = sp.csr_matrix(sm_).dot( sp.csr_matrix(sm_.T) ).toarray() > 0
sms = sp.csr_matrix(sms)
sp.save_npz("./mouse_human/sms.npz", sms)

smh = sp.csr_matrix(sm_).dot(sp.csr_matrix(mh_)) > 0
smhv = smh.dot(sp.csr_matrix(vh_.T)) > 0
smhvhms = smhv.dot(smh.T) > 0
smhvhms = sp.coo_matrix(smhvhms)
sp.save_npz("./mouse_human/smhvhms.npz", smhvhms)

#smh = np.matmul(sm_, mh_) > 0
#smhv = np.matmul(smh, vh_.T) > 0
smhvvhms = smhv.dot(smhv.T) > 0
smhvvhms = sp.coo_matrix(smhvvhms)
sp.save_npz("./mouse_human/smhvvhms.npz", smhvvhms)
'''





'''
pa = np.genfromtxt("./dblp/pa.txt")
pc = np.genfromtxt("./dblp/pc.txt")
pt = np.genfromtxt("./dblp/pt.txt")

A = 4057
P = 14328
C = 20
T = 7723

pa_ = sp.coo_matrix((np.ones(pa.shape[0]),(pa[:,0], pa[:, 1])),shape=(P,A)).toarray()
print(pa_.shape)
pc_ = sp.coo_matrix((np.ones(pc.shape[0]),(pc[:,0], pc[:, 1])),shape=(P,C)).toarray()
print(pc_.shape)
pt_ = sp.coo_matrix((np.ones(pt.shape[0]),(pt[:,0], pt[:, 1])),shape=(P,T)).toarray()
print(pt_.shape)


apa = np.matmul(pa_.T, pa_) > 0
apa = sp.coo_matrix(apa)
sp.save_npz("./dblp/apa.npz", apa)

apc = np.matmul(pa_.T, pc_) > 0
apcpa = np.matmul(apc, apc.T) > 0
apcpa = sp.coo_matrix(apcpa)
sp.save_npz("./dblp/apcpa.npz", apcpa)

apt = np.matmul(pa_.T, pt_) > 0
aptpa = np.matmul(apt, apt.T) > 0
aptpa = sp.coo_matrix(aptpa)
sp.save_npz("./dblp/aptpa.npz", aptpa)
'''