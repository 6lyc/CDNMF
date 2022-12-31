'''
simple mode
对比代码：去偏对比学习
A对比X
'''

import os
import pickle
import random
import torch
import gc
import torch.nn.functional as F
from sklearn.metrics.pairwise import rbf_kernel

class Model(torch.nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.device = config['device']
        self.net_shape = config['net_shape']
        self.att_shape = config['att_shape']
        self.net_input_dim = config['net_input_dim']
        self.att_input_dim = config['att_input_dim']
        self.is_init = config['is_init']
        self.pretrain_params_path = config['pretrain_params_path']
        self.tau = config['tau']
        self.conc = config['conc']
        self.negc = config['negc']
        self.rec = config['rec']
        self.r = config['r']
        self.model_path = config['model_path']

        self.fc1 = torch.nn.Linear(self.net_shape[-1], self.net_shape[1])
        self.fc2 = torch.nn.Linear(self.net_shape[1], self.net_shape[0])

        self.fc3 = torch.nn.Linear(self.att_shape[-1], self.net_shape[1])
        self.fc4 = torch.nn.Linear(self.net_shape[1], self.net_shape[0])

        self.U = torch.nn.ParameterDict({})
        self.V = torch.nn.ParameterDict({})

        # 读取时运行了其他code，同名文件被覆盖了
        if os.path.isfile(self.pretrain_params_path):
            with open(self.pretrain_params_path, 'rb') as handle:
                self.U_init, self.V_init = pickle.load(handle)

        if self.is_init:
            # 自定义可训练参数
            module = 'net'
            # print(len(self.net_shape))
            for i in range(len(self.net_shape)):
                name = module + str(i)
                self.U[name] = torch.nn.Parameter(torch.tensor(self.U_init[name], dtype=torch.float32))
            self.V[name] = torch.nn.Parameter(torch.tensor(self.V_init[name], dtype=torch.float32))

            module = 'att'
            # print(len(self.att_shape))
            for i in range(len(self.att_shape)):
                name = module + str(i)
                self.U[name] = torch.nn.Parameter(torch.tensor(self.U_init[name], dtype=torch.float32))
            self.V[name] = torch.nn.Parameter(torch.tensor(self.V_init[name], dtype=torch.float32))
        else:
            module = 'net'
            # print(len(self.net_shape))
            for i in range(len(self.net_shape)):
                name = module + str(i)
                self.U[name] = torch.nn.Parameter(torch.rand_like(torch.tensor(self.U_init[name], dtype=torch.float32)))
            self.V[name] = torch.nn.Parameter(torch.rand_like(torch.tensor(self.V_init[name], dtype=torch.float32)))

            module = 'att'
            # print(len(self.att_shape_shape))
            for i in range(len(self.att_shape)):
                name = module + str(i)
                self.U[name] = torch.nn.Parameter(torch.rand_like(torch.tensor(self.U_init[name]), dtype=torch.float32))
            self.V[name] = torch.nn.Parameter(torch.rand_like(torch.tensor(self.V_init[name]), dtype=torch.float32))

    def projection1(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z.t()))
        return self.fc2(z)

    def projection2(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc3(z.t()))
        return self.fc4(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        self.index_net = torch.argmax(self.V1, dim=0).long()
        O_net = F.one_hot(self.index_net, self.net_shape[-1]).float()
        S_net = torch.mm(O_net, O_net.t())
        # self.index_att = torch.argmax(self.V2, dim=0).long()
        # O_att = F.one_hot(self.index_att, self.att_shape[-1]).float()
        # S_att = torch.mm(O_att, O_att.t())

        refl_pos = refl_sim * S_net
        # between_pos = between_sim * S_net

        # return -torch.log(between_sim.diag() / (refl_sim.sum(1) - refl_pos.sum(1) + between_sim.sum(1) - between_pos.sum(1) + between_sim.diag()))
        return -torch.log((between_sim.diag()) / (refl_sim.sum(1) - refl_pos.sum(1) + between_sim.diag()))



    def contra_loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True):
        h1 = self.projection1(z1)
        h2 = self.projection2(z2)

        ret = self.semi_loss(h1, h2)
        # l2 = self.semi_loss(h2, h1)

        # ret = (l1 + l2) * 0.5
        # ret = l1
        ret = ret.mean() if mean else ret.sum()

        return ret

    def forward(self):
        # self.V1 = F.normalize(self.V['net' + str(len(self.net_shape) - 1)], p=2, dim=0)
        # self.V2 = F.normalize(self.V['att' + str(len(self.att_shape) - 1)], p=2, dim=0)
        self.V1 = self.V['net' + str(len(self.net_shape) - 1)]
        self.V2 = self.V['att' + str(len(self.att_shape) - 1)]
        # print('V1:', self.V1.shape)
        # print('V2:', self.V2)
        # return 0.5 * self.V1 + 0.5 * self.V2
        return self.V1

    def loss(self, graph):

        A = graph.A
        X = graph.X.T
        # D = torch.diag(torch.sum(graph.A, dim=1))
        # D_S = torch.diag(torch.sum(graph.A, dim=1).pow(-0.5))
        # L = torch.mm(torch.mm(D_S, D - graph.A), D_S)
        # print(L)
        L = torch.diag(torch.sum(graph.A, dim=1)) - A
        # XS = torch.from_numpy(rbf_kernel(graph.X.cpu())).to(self.device)
        # DX = torch.diag(torch.sum(XS, dim=1))
        # DX_S = torch.diag(torch.sum(XS, dim=1).pow(-0.5))
        # LX = torch.mm(torch.mm(DX_S, DX - XS), DX_S)
        # print(LX)
        # LX = F.normalize(DX - XS)

        # reconstruction
        P1 = torch.eye(self.net_input_dim, device=self.device)
        # print(P1)
        # print(self.U['net0'])
        for i in range(len(self.net_shape)):
            P1 = torch.mm(P1, self.U['net' + str(i)])
        i = len(self.net_shape) - 1
        P1 = torch.mm(P1, self.V['net' + str(i)])
        loss1 = torch.square(torch.norm(A - P1))
        # print('done:loss1', loss1)

        P2 = torch.eye(self.att_input_dim, device=self.device)
        for i in range(len(self.att_shape)):
            P2 = torch.mm(P2, self.U['att' + str(i)])
        i = len(self.att_shape) - 1
        P2 = torch.mm(P2, self.V['att' + str(i)])
        loss2 = torch.square(torch.norm(X - P2))
        # print('done:loss2', loss2)

        # contrastive loss
        # 不一定每次都更新
        loss3 = self.contra_loss(self.V1, self.V2)
        # print('done:loss3', loss3)

        # regularization loss
        i = len(self.net_shape) - 1
        # print(L)
        # print(self.V['net' + str(i)])
        # T = torch.mm(self.V['net' + str(i)], L)
        M = torch.mm(torch.mm(self.V['net' + str(i)], L), self.V['net' + str(i)].t())
        i = len(self.att_shape) - 1
        # print(L)
        # print(self.V['net' + str(i)])
        # TX = torch.mm(self.V['att' + str(i)], L)
        MX = torch.mm(torch.mm(self.V['att' + str(i)], L), self.V['att' + str(i)].t())
        loss4 = torch.trace(M) + torch.trace(MX)
        # loss4 = torch.trace(M)
        # print('done:loss4', loss4)

        # nonnegative loss item
        loss5 = 0
        for i in range(len(self.net_shape)):
            zero1 = torch.zeros_like(self.U['net' + str(i)])
            X1 = torch.where(self.U['net' + str(i)] > 0, zero1, self.U['net' + str(i)])
            loss5 = loss5 + torch.square(torch.norm(X1))
        zero1 = torch.zeros_like(self.V['net' + str(i)])
        X1 = torch.where(self.V['net' + str(i)] > 0, zero1, self.V['net' + str(i)])
        loss5 = loss5 + torch.square(torch.norm(X1))

        for i in range(len(self.att_shape)):
            zero2 = torch.zeros_like(self.U['att' + str(i)])
            X2 = torch.where(self.U['att' + str(i)] > 0, zero2, self.U['att' + str(i)])
            loss5 = loss5 + torch.square(torch.norm(X2))
        i = len(self.att_shape) - 1
        zero2 = torch.zeros_like(self.V['att' + str(i)])
        X2 = torch.where(self.V['att' + str(i)] > 0, zero2, self.V['att' + str(i)])
        loss5 = loss5 + torch.square(torch.norm(X2))
        # loss5 = -(torch.sum(X1) + torch.sum(X2))
        # print('done:loss5', loss5)

        loss = self.rec*(loss1 + loss2) + self.conc*loss3 + self.r*loss4 + self.negc*loss5
        # print('done:loss', loss)

        # for i in range(len(self.net_shape)):
        #     print(self. U['net' + str(i)].shape)


        return loss, loss1, loss2, loss3, loss4, loss5











