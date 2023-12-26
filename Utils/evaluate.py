import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score

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
'''




