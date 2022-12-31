import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score

'''  利用Python实现NMI、ACC、ARI计算'''

def NMI(A, B):
    # 样本点数
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    # 互信息计算
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(A == idA)  # 输出满足条件的元素的下标
            idBOccur = np.where(B == idB)
            idABOccur = np.intersect1d(idAOccur, idBOccur)  # Find the intersection of two arrays.
            px = 1.0 * len(idAOccur[0]) / total
            py = 1.0 * len(idBOccur[0]) / total
            pxy = 1.0 * len(idABOccur) / total
            MI = MI + pxy * math.log(pxy / (px * py) + eps, 2)
    # 标准化互信息
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0 * len(np.where(A == idA)[0])
        Hx = Hx - (idAOccurCount / total) * math.log(idAOccurCount / total + eps, 2)
    Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0 * len(np.where(B == idB)[0])
        Hy = Hy - (idBOccurCount / total) * math.log(idBOccurCount / total + eps, 2)
    MIhat = 2.0 * MI / (Hx + Hy)
    return MIhat


def MI(A, B):
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    # 互信息计算
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(A == idA)  # 输出满足条件的元素的下标
            idBOccur = np.where(B == idB)
            idABOccur = np.intersect1d(idAOccur, idBOccur)  # Find the intersection of two arrays.
            px = 1.0 * len(idAOccur[0]) / total
            py = 1.0 * len(idBOccur[0]) / total
            pxy = 1.0 * len(idABOccur) / total
            MI = MI + pxy * math.log(pxy / (px * py) + eps, 2)
    return MI


def bestMap(needmodified, ref, class_num):
    # 要求下标从0开始,长度相等，类别数相等
    if len(needmodified) != len(ref):
        print('需要对齐长度')
        return
    # 计算cost矩阵
    N = len(needmodified)

    K = class_num
    C = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            cij = 0
            for n in range(N):
                if needmodified[n] == i and ref[n] == j:
                    cij += 1
            C[i][j] = cij
    # 找最大匹配
    # print(C)
    row_ind, col_ind = linear_sum_assignment(C, maximize=True)
    # print(col_ind)

    modified_pred = np.zeros(N)
    for i in range(N):
        modified_pred[i] = col_ind[needmodified[i]]
    # 得到对应的类标

    return modified_pred.astype(int)


# def acc(pred,y_true,class_num):
#     modified_pred = bestMap(pred,y_true,class_num)
#     acc = np.sum(modified_pred==y_true)/len(modified_pred)
#     return acc


def cluster_acc(y_pred, y_true):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max().astype(int), y_true.max().astype(int)) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    # ind = sklearn.utils.linear_assignment_.linear_assignment(w.max() - w)
    # row_ind, col_ind = linear_assignment(w.max() - w)
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size


def clusterscores(pred: np.array, target: np.array):
    ACC = cluster_acc(pred, target)
    NMI = normalized_mutual_info_score(target, pred)
    ARI = adjusted_rand_score(target, pred)
    AMI = adjusted_mutual_info_score(target, pred)
    return {'ACC': ACC, 'NMI': NMI, 'ARI': ARI, 'AMI': AMI}

'''
if __name__ == "__main__":
    A = np.array([1, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 0, 1, 1, 0, 0, 0])
    # A = [1,2,1,1,1,1,1,2,2,2,2,0,1,1,0,0,0]
    B = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0])
    C = np.array([2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    D = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0])
    print(clusterscores(C, D)['ACC'])
    # print(NMI(A,C))
    # print(clusterscores(d,))
    # print(acc(C,A,3))
    # print(acc(A,D))
'''




