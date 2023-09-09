import os
import random
import numpy as np
import linecache
import matplotlib.pyplot as plt
import torch
from time import perf_counter as t
from Utils.evaluate import clusterscores
from Dataset.dataset import Dataset
from Model.my_model import Model
from PreTrainer.pretrainer import PreTrainer
from Utils import gpu_info

max_acc = 0

def train(model: Model, graph, optimizer):
    optimizer.zero_grad()
    V = model()

    loss, loss1, loss2, loss3, loss4, loss5 = model.loss(graph)
    loss.backward()
    optimizer.step()

    y_pred = np.argmax(V.detach().cpu().numpy(), axis=0)
    y_true = graph.L.detach().cpu().numpy()
    # print(y_pred)
    scores = clusterscores(y_pred, y_true)
    global max_acc
    if scores['ACC'] > max_acc:
        max_acc = scores['ACC']

    return loss.item(), loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item(), scores


if __name__=='__main__':

    random.seed(9001)

    dataset_config = {'feature_file': './Database/cora/features.txt',
                      'graph_file': './Database/cora/edges.txt',
                      'walks_file': './Database/cora/walks.txt',
                      'label_file': './Database/cora/group.txt'}
    graph = Dataset(dataset_config)

    pretrain_config = {
        'net_shape': [256, 64, 7],
        'att_shape': [200, 100, 7],
        'net_input_dim': graph.num_nodes,
        'att_input_dim': graph.num_feas,
        'seed': 21,
        'pre_iterations': 100,
        'pretrain_params_path': './Log/cora/pretrain_params.pkl'}

    model_config = {
        'net_shape': [256, 64, 7],
        'att_shape': [200, 100, 7],
        'net_input_dim': graph.num_nodes,
        'att_input_dim': graph.num_feas,
        'is_init': True,
        'pretrain_params_path': './Log/cora/pretrain_params.pkl',
        'tau': 0.5,
        'conc': 1,
        'negc': 100,
        'rec': 1,
        'learning_rate': 0.01,
        'weight_decay': 0.00001,
        'model_path': './Log/cora/cora_model.pkl'
    }

    # pretrainer = PreTrainer(pretrain_config)
    # pretrainer.pre_training(graph.A.detach().cpu().numpy(), 'net')

    # 可以用谱聚类中的相似矩阵重构邻接矩阵（包含全局的信息）
    # print(graph.A.shape)
    # print(graph.X.shape)
    # pretrainer.pre_training(graph.X.t().detach().cpu().numpy(), 'att')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    '''
    
    for param in model.parameters():
        print(type(param.data), param.size())
    '''


    learning_rate = model_config['learning_rate']
    weight_decay = model_config['weight_decay']


    start = t()
    prev = start

    X = []
    Y = []
    Y1 = []
    Y2 = []
    Y3 = []
    Y4 = []
    Y5 = []
    A = []
    N = []
    R = []
    M = []
    for i in range(5):
        max_acc = 0
        model = Model(model_config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(1, 201):
            loss, loss1, loss2, loss3, loss4, loss5, scores = train(model, graph, optimizer)

            X.append(epoch)
            Y.append(loss)
            Y1.append(loss1)
            Y2.append(loss2)
            Y3.append(loss3)
            Y4.append(loss4)
            Y5.append(loss5)
            A.append(scores['ACC'])
            N.append(scores['NMI'])
            R.append(scores['ARI'])

            now = t()
            # print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, accuracy={scores} max_accuracy={max_acc:4f} ')
            # print(f'this epoch {now - prev:.4f}, total {now - start:.4f}')
            prev = now
        print(f'max_accuracy={max_acc:4f}')
        M.append(max_acc)
    print(np.mean(M))
    print("=== Final ===")
'''
    plt.figure(1)
    plt.subplot(231)
    plt.plot(X, Y)

    plt.subplot(232)
    plt.plot(X, Y1)

    plt.subplot(233)
    plt.plot(X, Y2)

    plt.subplot(234)
    plt.plot(X, Y3)

    plt.subplot(235)
    plt.plot(X, Y4)

    plt.subplot(236)
    plt.plot(X, Y5)

    plt.figure(2)
    plt.subplot(221)
    plt.plot(X, A)

    plt.subplot(222)
    plt.plot(X, N)

    plt.subplot(223)
    plt.plot(X, R)

    plt.show()

'''







