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

    return loss.item(), loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item(), scores


if __name__=='__main__':

    random.seed(42)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset_config = {'feature_file': './Database/citeseer/features.txt',
                      'graph_file': './Database/citeseer/edges.txt',
                      'walks_file': './Database/citeseer/walks.txt',
                      'label_file': './Database/citeseer/group.txt',
                      'device': device}
    graph = Dataset(dataset_config)

    pretrain_config = {
        'net_shape': [256, 64, 6],
        'att_shape': [500, 100, 6],
        'net_input_dim': graph.num_nodes,
        'att_input_dim': graph.num_feas,
        'seed': 42,
        'pre_iterations': 100,
        'pretrain_params_path': './Log/citeseer/pretrain_params.pkl'}

    model_config = {
        'device': device,
        'net_shape': [256, 64, 6],
        'att_shape': [500, 100, 6],
        'net_input_dim': graph.num_nodes,
        'att_input_dim': graph.num_feas,
        'is_init': True,
        'pretrain_params_path': './Log/citeseer/pretrain_params.pkl',
        'tau': 1.5,
        'conc': 5,
        'negc': 2900,
        'rec': 1,
        'r': 1,
        'learning_rate': 0.001,
        'weight_decay': 0.00001,
        'epoch': 1000,
        'run': 20,
        'model_path': './Log/citeseer/citeseer_model.pkl'
    }

    # 'Pre-training stage'
    # pretrainer = PreTrainer(pretrain_config)
    # pretrainer.pre_training(graph.A.detach().cpu().numpy(), 'net')
    # pretrainer.pre_training(graph.X.t().detach().cpu().numpy(), 'att')


    learning_rate = model_config['learning_rate']
    weight_decay = model_config['weight_decay']


    start = t()
    prev = start

    M = []
    N = []
    
    # 'Fine-tuning stage'
    for i in range(20):

        model = Model(model_config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(1, model_config['epoch']):
            loss, loss1, loss2, loss3, loss4, loss5, scores = train(model, graph, optimizer)

            now = t()
            prev = now

        M.append(scores['ACC'])
        N.append(scores['NMI'])
        
    print('ACC: ', np.mean(M), '; NMI: ', np.mean(N))
    print("=== Final ===")





