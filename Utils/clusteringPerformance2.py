'''
   This program is to evaluate clustering performance

   Code author: Shide Du
   Email: shidedums@163.com
   Date: Dec 4, 2019.
'''
import torch
import torch.nn.functional as F
from scipy.stats import mode
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
import numpy.linalg as LA
from sklearn import metrics
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score
# from sklearn.metrics.cluster.supervised import check_clusterings
# from sklearn.metrics.cluster.supervised import check_clusterings
import warnings
from scipy import sparse as sp
# from sklearn.utils.fixes import comb
from scipy.special import comb
from sklearn.preprocessing import normalize
warnings.filterwarnings("ignore")

def bestMap(y_pred,y_true):
    from scipy.optimize import linear_sum_assignment
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    np.asarray(ind)
    ind = np.transpose(ind)
    label=np.zeros(y_pred.size)
    for i in range(y_pred.size):
        label[i]=ind[y_pred[i]][1]
    return label.astype(np.int64)


### K-means clustering
def KMeansClustering(features, gnd, clusterNum, randNum):
    kmeans = KMeans(n_clusters=clusterNum, n_init=1, max_iter=500,
                    random_state=randNum)
    estimator = kmeans.fit(features)
    clusters = estimator.labels_
    #sio.savemat('ALOI_idx1.mat', {'idx': label_pred})
    # sio.savemat('Caltech101-7_idx1.mat', {'idx': label_pred})
    # 获取聚类标签
    #centroids = estimator.cluster_centers_  # 获取聚类中心
    #inertia = estimator.inertia_  # 获取聚类准则的总和
    #X_tsne = TSNE(n_components=2, learning_rate=100,  random_state=0).fit_transform(features)
    #plt.figure(figsize=(12, 6))

    #plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=label_pred)
    #plt.colorbar()
    #plt.show()

######################################################
    #plt.xlim((-70, 70))
    #plt.ylim((-70, 70))
    #kmeans.fit(features)
    #tsne = TSNE(n_components=2, learning_rate=100,  random_state=0)
    # 对数据进行降维
    #tsne.fit_transform(features)
    #data = pd.DataFrame(tsne.embedding_, index=label_pred)

    #d = data[kmeans.labels_ == 0]
    #plt.plot(d[0], d[1],  'ro', label = '1')
    #d = data[kmeans.labels_ == 1]
    #plt.plot(d[0], d[1], 'go', label = '2')
    #d = data[kmeans.labels_ == 2]
    #plt.plot(d[0], d[1], 'bo', label = '3')
    #d = data[kmeans.labels_ == 3]
    #plt.plot(d[0], d[1], 'yo', label = '4')
    #d = data[kmeans.labels_ == 4]
    #plt.plot(d[0], d[1], 'ko', label='5')
    #d = data[kmeans.labels_ == 5]
    #plt.plot(d[0], d[1], 'mo', label='6')
    #d = data[kmeans.labels_ == 6]
    #plt.plot(d[0], d[1], 'co', label='7')
    #plt.legend(loc='best')
    #plt.savefig("Caltech101-7NEWDBMC.svg")
    #plt.show()
#########################################################################################33
    # 不同类别用不同颜色和样式绘图
    #d = data[kmeans.labels_ == 0]
    #plt.scatter(d[0], d[1], label = '1', s = 3, color='r', marker='.')
    #d = data[kmeans.labels_ == 1]
    #plt.scatter(d[0], d[1], label = '2', s = 3, color='g', marker='o')
    #d = data[kmeans.labels_ == 2]
    #plt.scatter(d[0], d[1], label = '3', s = 3, color='lawngreen', marker='*')
    #d = data[kmeans.labels_ == 3]
    #plt.scatter(d[0], d[1], label = '4', s = 3, color='b', marker='4')
    #d = data[kmeans.labels_ == 4]
    #plt.scatter(d[0], d[1], label='5', s = 3, color='y', marker='1')
    #d = data[kmeans.labels_ == 5]
    #plt.scatter(d[0], d[1], label='6', s = 3, color='chartreuse', marker='2')
    #d = data[kmeans.labels_ == 6]
    #plt.scatter(d[0], d[1], label='7', s = 3, color='deepskyblue', marker='3')
    #d = data[kmeans.labels_ == 7]
    #plt.scatter(d[0], d[1], label='8', s=2, color='skyblue', marker='>')
    #plt.axis('off')
    #plt.legend(loc='best')
    #plt.savefig("Caltech101-7NEWDBMC.svg")
    #plt.show()

    # print("The type of clusters is: ", type(clusters))
    # print("Clustering results are: ", clusters.shape)

    labels = np.zeros_like(clusters)
    for i in range(clusterNum):
        mask = (clusters == i)
        labels[mask] = mode(gnd[mask])[0]
    #sio.savemat('ALOI_idx.mat', {'idx': labels})
    # Return the preditive clustering label
    return labels


def similarity_function(points):
    """

    :param points:
    :return:
    """
    res = rbf_kernel(points)
    for i in range(len(res)):
        res[i, i] = 0
    return res

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    # ind = linear_assignment(w.max() - w)
    indx_list = []
    for i in range(len(ind[0])):
        indx_list.append((ind[0][i], ind[1][i]))
    # return sum([w[i1, j1] for i1, j1 in ind]) * 1.0 / y_pred.size
    return sum([w[i1, j1] for (i1, j1) in indx_list]) * 1.0 / y_pred.size


def cluster_f(y_true, y_pred):
    N = len(y_true)
    numT = 0
    numH = 0
    numI = 0
    for n in range(0, N):
        C1 = [y_true[n] for x in range(1, N - n)]
        C1 = np.array(C1)
        C2 = y_true[n + 1:]
        C2 = np.array(C2)
        Tn = (C1 == C2)*1

        C3 = [y_pred[n] for x in range(1, N - n)]
        C3 = np.array(C3)
        C4 = y_pred[n + 1:]
        C4 = np.array(C4)
        Hn = (C3 == C4)*1

        numT = numT + np.sum(Tn)
        numH = numH + np.sum(Hn)
        numI = numI + np.sum(np.multiply(Tn, Hn))
    if numH > 0:
        p = numI / numH
    if numT > 0:
        r = numI / numT
    if (p + r) == 0:
        f = 0;
    else:
        f = 2 * p * r / (p + r);
    return f, p, r


def clustering_purity(labels_true, labels_pred):
    """
    :param y_true:
        data type: numpy.ndarray
        shape: (n_samples,)
        sample: [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20]
    :param y_pred:
        data type: numpy.ndarray
        shape: (n_samples,)
        sample: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
    :return: Purity
    """
    y_true = labels_true.copy()
    y_pred = labels_pred.copy()
    if y_true.shape[1] != 1:
        y_true = y_true.T
    if y_pred.shape[1] != 1:
        y_pred = y_pred.T

    n_samples = len(y_true)

    u_y_true = np.unique(y_true)
    n_true_classes = len(u_y_true)
    y_true_temp = np.zeros((n_samples, 1))
    if n_true_classes != max(y_true):
        for i in range(n_true_classes):
            y_true_temp[np.where(y_true == u_y_true[i])] = i + 1
        y_true = y_true_temp

    u_y_pred = np.unique(y_pred)
    n_pred_classes = len(u_y_pred)
    y_pred_temp = np.zeros((n_samples, 1))
    if n_pred_classes != max(y_pred):
        for i in range(n_pred_classes):
            y_pred_temp[np.where(y_pred == u_y_pred[i])] = i + 1
        y_pred = y_pred_temp

    u_y_true = np.unique(y_true)
    n_true_classes = len(u_y_true)
    u_y_pred = np.unique(y_pred)
    n_pred_classes = len(u_y_pred)

    n_correct = 0
    for i in range(n_pred_classes):
        incluster = y_true[np.where(y_pred == u_y_pred[i])]

        inclunub = np.histogram(incluster, bins = range(1, int(max(incluster)) + 1))[0]
        if len(inclunub) != 0:
            n_correct = n_correct + max(inclunub)

    Purity = n_correct/len(y_pred)

    return Purity

def contingency_matrix(labels_true, labels_pred, eps=None, sparse=False):
    """Build a contingency matrix describing the relationship between labels.

    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        Ground truth class labels to be used as a reference

    labels_pred : array, shape = [n_samples]
        Cluster labels to evaluate

    eps : None or float, optional.
        If a float, that value is added to all values in the contingency
        matrix. This helps to stop NaN propagation.
        If ``None``, nothing is adjusted.

    sparse : boolean, optional.
        If True, return a sparse CSR continency matrix. If ``eps is not None``,
        and ``sparse is True``, will throw ValueError.

        .. versionadded:: 0.18

    Returns
    -------
    contingency : {array-like, sparse}, shape=[n_classes_true, n_classes_pred]
        Matrix :math:`C` such that :math:`C_{i, j}` is the number of samples in
        true class :math:`i` and in predicted class :math:`j`. If
        ``eps is None``, the dtype of this array will be integer. If ``eps`` is
        given, the dtype will be float.
        Will be a ``scipy.sparse.csr_matrix`` if ``sparse=True``.
    """

    if eps is not None and sparse:
        raise ValueError("Cannot set 'eps' when sparse=True")

    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]
    # Using coo_matrix to accelerate simple histogram calculation,
    # i.e. bins are consecutive integers
    # Currently, coo_matrix is faster than histogram2d for simple cases
    contingency = sp.coo_matrix((np.ones(class_idx.shape[0]),
                                 (class_idx, cluster_idx)),
                                shape=(n_classes, n_clusters),
                                dtype=np.int)
    if sparse:
        contingency = contingency.tocsr()
        contingency.sum_duplicates()
    else:
        contingency = contingency.toarray()
        if eps is not None:
            # don't use += as contingency is integer
            contingency = contingency + eps
    return contingency

def _comb2(n):
    # the exact version is faster for k == 2: use it by default globally in
    # this module instead of the float approximate variant
    return comb(n, 2, exact=1)

def rand_index_score(labels_true, labels_pred):
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    n_samples = labels_true.shape[0]
    n_classes = np.unique(labels_true).shape[0]
    n_clusters = np.unique(labels_pred).shape[0]

    # Special limit cases: no clustering since the data is not split;
    # or trivial clustering where each document is assigned a unique cluster.
    # These are perfect matches hence return 1.0.
    if (n_classes == n_clusters == 1 or
            n_classes == n_clusters == 0 or
            n_classes == n_clusters == n_samples):
        return 1.0

    # Compute the RI using the contingency data
    contingency = contingency_matrix(labels_true, labels_pred)

    n = np.sum(np.sum(contingency))
    t1 = comb(n, 2)
    t2 = np.sum(np.sum(np.power(contingency, 2)))
    nis = np.sum(np.power(np.sum(contingency, 0), 2))
    njs = np.sum(np.power(np.sum(contingency, 1), 2))
    t3 = 0.5 * (nis + njs)

    A = t1 + t2 - t3
    nc = (n * (n ** 2 + 1) - (n + 1) * nis - (n + 1) * njs + 2 * (nis * njs) / n) / (2 * (n - 1))
    AR = (A - nc) / (t1 - nc)
    return A / t1

def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment as linear_assignment
    # from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def b3_precision_recall_fscore(labels_true, labels_pred):
    """Compute the B^3 variant of precision, recall and F-score.
    Parameters
    ----------
    :param labels_true: 1d array containing the ground truth cluster labels.
    :param labels_pred: 1d array containing the predicted cluster labels.
    Returns
    -------
    :return float precision: calculated precision
    :return float recall: calculated recall
    :return float f_score: calculated f_score
    Reference
    ---------
    Amigo, Enrique, et al. "A comparison of extrinsic clustering evaluation
    metrics based on formal constraints." Information retrieval 12.4
    (2009): 461-486.
    """
    # Check that labels_* are 1d arrays and have the same size

    labels_pred = bestMap(labels_pred, labels_true)

    # Check that input given is not the empty set
    if labels_true.shape == (0,):
        raise ValueError(
            "input labels must not be empty.")

    # Compute P/R/F scores
    n_samples = len(labels_true)
    true_clusters = {}  # true cluster_id => set of sample indices
    pred_clusters = {}  # pred cluster_id => set of sample indices

    for i in range(n_samples):
        true_cluster_id = labels_true[i]
        pred_cluster_id = labels_pred[i]

        if true_cluster_id not in true_clusters:
            true_clusters[true_cluster_id] = set()
        if pred_cluster_id not in pred_clusters:
            pred_clusters[pred_cluster_id] = set()

        true_clusters[true_cluster_id].add(i)
        pred_clusters[pred_cluster_id].add(i)

    for cluster_id, cluster in true_clusters.items():
        true_clusters[cluster_id] = frozenset(cluster)
    for cluster_id, cluster in pred_clusters.items():
        pred_clusters[cluster_id] = frozenset(cluster)

    precision = 0.0
    recall = 0.0

    intersections = {}

    for i in range(n_samples):
        pred_cluster_i = pred_clusters[labels_pred[i]]
        true_cluster_i = true_clusters[labels_true[i]]

        if (pred_cluster_i, true_cluster_i) in intersections:
            intersection = intersections[(pred_cluster_i, true_cluster_i)]
        else:
            intersection = pred_cluster_i.intersection(true_cluster_i)
            intersections[(pred_cluster_i, true_cluster_i)] = intersection

        precision += len(intersection) / len(pred_cluster_i)
        recall += len(intersection) / len(true_cluster_i)

    precision /= n_samples
    recall /= n_samples

    f_score = 2 * precision * recall / (precision + recall)

    return f_score, precision, recall

### Evaluation metrics of clustering performance
def clusteringMetrics(trueLabel, predictiveLabel):
    y_pred=bestMap(predictiveLabel,trueLabel)
    # Clustering accuracy
    ACC = cluster_acc(trueLabel, y_pred)

    # Normalized mutual information
    #NMI = metrics.v_measure_score(trueLabel, predictiveLabel)
    NMI = normalized_mutual_info_score(trueLabel, y_pred)

    # Purity
    Purity = clustering_purity(trueLabel.reshape((-1, 1)), y_pred.reshape(-1, 1))

    # Adjusted rand index
    ARI = metrics.adjusted_rand_score(trueLabel, y_pred)
    #ARI = rand_index_score(trueLabel, predictiveLabel)

    # Fscore, Precision, Recall = cluster_f(trueLabel, y_pred)
    Fscore, Precision, Recall = b3_precision_recall_fscore(trueLabel, y_pred)

    # return ACC, NMI, Purity, ARI, Fscore, Precision, Recall
    return {'ACC': ACC, 'NMI': NMI, 'Purity': Purity, 'ARI': ARI, 'Fscore': Fscore, 'Precision': Precision, 'Recall': Recall}


### Report mean and std of 10 experiments
def StatisticClustering(features, gnd, clusterNum):
    ### Input the mean and standard diviation with 10 experiments
    repNum = 10
    ACCList = np.zeros((repNum, 1))
    NMIList = np.zeros((repNum, 1))
    PurityList = np.zeros((repNum, 1))
    ARIList = np.zeros((repNum, 1))
    FscoreList = np.zeros((repNum, 1))
    PrecisionList = np.zeros((repNum, 1))
    RecallList = np.zeros((repNum, 1))

    #clusterNum = int(np.max(gnd)) - int(np.min(gnd)) + 1
    # print("cluster number: ", clusterNum)
    for i in range(repNum):
        predictiveLabel = KMeansClustering(features, gnd, clusterNum, i)
        ACC, NMI, Purity, ARI, Fscore, Precision, Recall = clusteringMetrics(gnd, predictiveLabel)

        ACCList[i] = ACC
        NMIList[i] = NMI
        PurityList[i] = Purity
        ARIList[i] = ARI
        FscoreList[i] = Fscore
        PrecisionList[i] = Precision
        RecallList[i] = Recall
        # print("ACC, NMI, ARI: ", ACC, NMI, ARI)
    ACCmean_std = np.around([np.mean(ACCList), np.std(ACCList)], decimals=4)
    NMImean_std = np.around([np.mean(NMIList), np.std(NMIList)], decimals=4)
    Puritymean_std = np.around([np.mean(PurityList), np.std(PurityList)], decimals=4)
    ARImean_std = np.around([np.mean(ARIList), np.std(ARIList)], decimals=4)
    Fscoremean_std = np.around([np.mean(FscoreList), np.std(FscoreList)], decimals=4)
    Precisionmean_std = np.around([np.mean(PrecisionList), np.std(PrecisionList)], decimals=4)
    Recallmean_std = np.around([np.mean(RecallList), np.std(RecallList)], decimals=4)
    #plt.scatter(features[:, 0], features[:, 2], c = predictiveLabel)
    #plt.savefig("Clustering_results.jpg")
    #plt.show()
    return ACCmean_std, NMImean_std, Puritymean_std, ARImean_std, Fscoremean_std, Precisionmean_std, Recallmean_std


def StatisticClustering1(features, gnd):
    ### Input the mean and standard diviation with 10 experiments
    repNum = 7
    ACCList = np.zeros((repNum, 1))
    NMIList = np.zeros((repNum, 1))
    ARIList = np.zeros((repNum, 1))
    clusterNum = int(np.max(gnd)) - int(np.min(gnd)) + 1
    print("cluster number: ", clusterNum)
    for i in range(repNum):
        predictiveLabel = KMeansClustering(features, gnd, clusterNum, i)
        ACC, NMI, ARI = clusteringMetrics(gnd, predictiveLabel)
        ACCList[i] = ACC
        NMIList[i] = NMI
        ARIList[i] = ARI
        # print("ACC, NMI, ARI: ", ACC, NMI, ARI)
    ACCmean_std = np.around([np.mean(ACCList), np.std(ACCList)], decimals=4)
    NMImean_std = np.around([np.mean(NMIList), np.std(NMIList)], decimals=4)
    ARImean_std = np.around([np.mean(ARIList), np.std(ARIList)], decimals=4)
    return ACCmean_std, NMImean_std, ARImean_std


def spectral_clustering(points, k, gnd):
    W = similarity_function(points)
    #W = points
    Dn = np.diag(1 / np.power(np.sum(W, axis=1), -0.5))
    L = np.eye(len(points)) - np.dot(np.dot(Dn, W), Dn)
    eigvals, eigvecs = LA.eig(L)
    eigvecs = eigvecs.astype(float)
    indices = np.argsort(eigvals)[:k]
    k_smallest_eigenvectors = normalize(eigvecs[:, indices])

    [ACC, NMI, Purity, ARI, Fscore, Precision, Recall] = StatisticClustering(k_smallest_eigenvectors, gnd, k)
    return [ACC, NMI, Purity, ARI, Fscore, Precision, Recall]


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def clustering_loss(features, gnd, clusterNum, alpha, device):
    ### Input the mean and standard diviation with 10 experiments
    repNum = 10
    ACCList = np.zeros((repNum, 1))
    NMIList = np.zeros((repNum, 1))
    PurityList = np.zeros((repNum, 1))
    ARIList = np.zeros((repNum, 1))
    FscoreList = np.zeros((repNum, 1))
    PrecisionList = np.zeros((repNum, 1))
    RecallList = np.zeros((repNum, 1))
    kl_loss = 0
    #clusterNum = int(np.max(gnd)) - int(np.min(gnd)) + 1
    # print("cluster number: ", clusterNum)
    for i in range(repNum):
        kmeans = KMeans(n_clusters=clusterNum)
        kmeans.fit_predict(features.cpu().detach().squeeze().numpy())
        cluster_layer = torch.tensor(kmeans.cluster_centers_).to(device)
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(features.squeeze().unsqueeze(1) - cluster_layer, 2), 2) / alpha)
        q = q.pow((alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        p = target_distribution(q)
        y_pred = q.cpu().detach().numpy().argmax(1)
        # delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
        # y_pred_last = y_pred
        ACC, NMI, Purity, ARI, Fscore, Precision, Recall = clusteringMetrics(trueLabel=gnd, predictiveLabel=y_pred)

        ACCList[i] = ACC
        NMIList[i] = NMI
        PurityList[i] = Purity
        ARIList[i] = ARI
        FscoreList[i] = Fscore
        PrecisionList[i] = Precision
        RecallList[i] = Recall
        kl_loss += F.kl_div(q.log(), p)
        # print("ACC, NMI, ARI: ", ACC, NMI, ARI)
    ACCmean_std = np.around([np.mean(ACCList), np.std(ACCList)], decimals=4)
    NMImean_std = np.around([np.mean(NMIList), np.std(NMIList)], decimals=4)
    Puritymean_std = np.around([np.mean(PurityList), np.std(PurityList)], decimals=4)
    ARImean_std = np.around([np.mean(ARIList), np.std(ARIList)], decimals=4)
    Fscoremean_std = np.around([np.mean(FscoreList), np.std(FscoreList)], decimals=4)
    Precisionmean_std = np.around([np.mean(PrecisionList), np.std(PrecisionList)], decimals=4)
    Recallmean_std = np.around([np.mean(RecallList), np.std(RecallList)], decimals=4)
    #plt.scatter(features[:, 0], features[:, 2], c = predictiveLabel)
    #plt.savefig("Clustering_results.jpg")
    #plt.show()
    return ACCmean_std, NMImean_std, Puritymean_std, ARImean_std, Fscoremean_std, Precisionmean_std, Recallmean_std, kl_loss

### Real entrance to this program
if __name__ == '__main__':
    # Step 1: load data
    #features, gnd = loadData('./data/Yale_32x32.mat')
    #print("The size of data matrix is: ", features.shape)
    #gnd = gnd.flatten()
    #print("The size of data label is: ", gnd.shape)
    #clusterNum = 10
    # Print clustering results
    #[ACCmean_std, NMImean_std, Puritymean_std, ARImean_std, Fscoremean_std, Precisionmean_std, Recallmean_std] = StatisticClustering(
    #    features, gnd)
    #print("ACC, NMI, Purity, ARI, Fscore, Precision, Recall: ", ACCmean_std, NMImean_std, Puritymean_std, ARImean_std, Fscoremean_std, Precisionmean_std, Recallmean_std)
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    gnd = np.array([1, 1, 1, 0, 0, 0])
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    estimator = kmeans.fit(X)
    clusters = estimator.labels_
    label_pred = estimator.labels_
    labels = np.zeros_like(clusters)
    for i in range(2):
        mask = (clusters == i)
        labels[mask] = mode(gnd[mask])[0]
    print(labels)
    print(label_pred)