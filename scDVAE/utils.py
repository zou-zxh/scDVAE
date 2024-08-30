#!/usr/bin/env python
"""
#
#

# File Name: utils.py
# Description:

"""


import numpy as np
import pandas as pd
import scipy
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, scale
from sklearn.metrics import classification_report, confusion_matrix, adjusted_rand_score
from scipy.optimize import linear_sum_assignment as linear_assignment

# ============== Data Processing ==============
# =============================================

def read_labels(ref, return_enc=False):
    """
    Read labels and encode to 0, 1 .. k with class names 
    """
    # if isinstance(ref, str):
    ref = pd.read_csv(ref, sep='\t', index_col=0, header=None)

    encode = LabelEncoder()
    ref = encode.fit_transform(ref.values.squeeze())
    classes = encode.classes_
    if return_enc:
        return ref, classes, encode
    else:
        return ref, classes



# =================== Other ====================
# ==============================================

def estimate_k(data):
    """
    Estimate number of groups k:
        based on random matrix theory (RTM), borrowed from SC3
        input data is (p,n) matrix, p is feature, n is sample（cell)
    """
    p, n = data.shape
    if type(data) is not np.ndarray:
        data = data.toarray()
    # scale函数对每一列的数据进行标准化
    # 其对应的公式可以在论文的 Select an appropriate number of clusters K for GMM sampling中看到
    x = scale(data)
    muTW = (np.sqrt(n-1) + np.sqrt(p)) ** 2
    sigmaTW = (np.sqrt(n-1) + np.sqrt(p)) * (1/np.sqrt(n-1) + 1/np.sqrt(p)) ** (1/3)
    sigmaHatNaive = x.T.dot(x)

    bd = np.sqrt(p) * sigmaTW + muTW
    evals = np.linalg.eigvalsh(sigmaHatNaive)  # 计算x^Tx矩阵的特征值

    k = 0
    for i in range(len(evals)):
        if evals[i] > bd:
            k += 1
    return k


def pairwise_pearson(A, B):
    from scipy.stats import pearsonr
    corrs = []
    for i in range(A.shape[0]):
        if A.shape == B.shape:
            corr = pearsonr(A.iloc[i], B.iloc[i])[0]
        else:
            corr = pearsonr(A.iloc[i], B)[0]
        corrs.append(corr)
    return corrs

# ================= Metrics ===================
# =============================================

def reassign_cluster_with_ref(Y_pred, Y):
    """
    Reassign cluster to reference labels
    Inputs:
        Y_pred: predict y classes
        Y: true y classes
    Return:
        f1_score: clustering f1 score
        y_pred: reassignment index predict y classes
        indices: classes assignment
    """
    def reassign_cluster(y_pred, index):
        y_ = np.zeros_like(y_pred)
        for i, j in index:
            y_[np.where(y_pred==i)] = j
        return y_
#     print(Y_pred.size, Y.size)
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)

    return reassign_cluster(Y_pred, ind)

def cluster_report(ref, pred, classes):
    """
    Print Cluster Report
    """
    pred = reassign_cluster_with_ref(pred, ref)
    cm = confusion_matrix(ref, pred)
    print('\n## Confusion matrix ##\n')
    print(cm)
    print('\n## Cluster Report ##\n')
    print(classification_report(ref, pred, target_names=classes))
    ari_score = adjusted_rand_score(ref, pred)
    print("\nAdjusted Rand score : {:.4f}".format(ari_score))



def cluster_acc(Y_pred,Y):
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(),Y.max())+1  # 因为类别的标签是从0开始的
    w = np.zeros((D,D),dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i],Y[i]] +=1 # Y_pred[i]只有10种类别，想要知道预测类别和真实类别的数据只要关注对角线的数据就好 w[0,0]表示两者都属于第0类
    """linear_assignment是一个分配函数，找到一种最佳的分配方式，使得不同的聚类标签之间的共同分配数量最大化
    会生成两个索引对，如生成ind:array([0,1,2,3,4,5,6,7,8,9]),array([9,3,0,6,1,2,7,5,4,8])
    表示将第一个聚类标签0被分配给第九个聚类簇，在训练之前只是知道有10个聚类标签，但是无法判断我们所聚类的细胞簇到底是属于哪个标签的"""
    ind = linear_assignment(w.max()-w)
    total = 0
    for i in range(len(ind[0])):
        total += w[ind[0][i],ind[1][i]]  # 首先根据我们制定聚类规则，取出预测标签和真实标签一致的数据，然后将其得到一致的总数，计算准确性
    return total*1.0/Y_pred.size,w
    # 返回一个准确率

def gmm_Loss(z, z_mu, z_sigma2_log,gmm):

    # GMM的损失函数目标是在训练过程中平衡数据似然性和潜在变量的稳定性，从而使模型能够学习到适合数据的潜在聚类结构
    det = 1e-10
    pi = gmm.pi_
    mu_c = gmm.mu_c
    log_sigma2_c = gmm.log_sigma2_c

    yita_c = torch.exp(torch.log(pi.unsqueeze(0)) + gmm.gaussian_pdfs_log(z,mu_c,log_sigma2_c)) +det

    yita_c = yita_c / (yita_c.sum(1).view(-1,1))

    Loss = 0.5 * torch.mean(torch.sum(yita_c * torch.sum(log_sigma2_c.unsqueeze(0)+
                                                         torch.exp(z_sigma2_log.unsqueeze(1)-log_sigma2_c.unsqueeze(0))+
                                                         (z_mu.unsqueeze(1)-mu_c.unsqueeze(0)).pow(2)/
                                                         torch.exp(log_sigma2_c.unsqueeze(0)),2),1)) # GMM的最大似然估计损失函数
    # a = torch.mean(torch.sum(yita_c * torch.log(pi.unsqueeze(0) / (yita_c)), 1))
    # b = 0.5 * torch.mean(torch.sum(1 + z_sigma2_log, 1))
    Loss -= torch.mean(torch.sum(yita_c * torch.log(pi.unsqueeze(0) / (yita_c)), 1)) + 0.5 * torch.mean(torch.sum(1 + z_sigma2_log, 1)) # kl散度损失
    return Loss


def mask_generator(p_m,x):
    """
    np.random.binomial生成一个与输入特征矩阵x具有相同形状的二进制掩码矩阵mask,会以特定的概率p_m在每个位置上生成值为1，0.3生成0
    """
    mask = np.random.binomial(1,p_m,x.shape)
    return  mask

def pretext_generator(m,x):
    """
    m是一个随机掩码矩阵，其中元素只有1或0两种
    70%元素是1，30%元素是0
    每一列都包含原始数据，但是存储时行的的顺序被打乱了，数据具有一定的随机性
    """
    no,dim = x.shape
    x_bar = np.zeros([no,dim])
    for i in range(dim):
        idx = np.random.permutation(no) # 生成的是一个随机索引组
        x_bar[:,i]=x_bar[idx,i]

    x_tilde = x * (1-m) + x_bar *m # 保留了30%的原始数据，70%的随机数据
    m_new = 1 * (x != x_tilde)  #两个矩阵中数据不相等，则m_new标记为1，相等则标记为0

    return m_new,x_tilde

