import numpy as np
import linecache
import torch


class Dataset(object):

    def __init__(self, config):
        self.graph_file = config['graph_file']
        self.feature_file = config['feature_file']
        self.label_file = config['label_file']
        self.walks_file = config['walks_file']
        self.device = config['device']

        self.A, self.X, self.W, self.L, self.num_classes = self._load_data()

        self.num_nodes = self.A.shape[0]
        self.num_feas = self.X.shape[1]
        self.num_edges = np.sum(self.A) / 2

        self.A = torch.tensor(self.A, dtype=torch.float32, device=self.device)
        self.X = torch.tensor(self.X, dtype=torch.float32, device=self.device)
        self.W = torch.tensor(self.W, dtype=torch.float32, device=self.device)
        self.L = torch.tensor(self.L, dtype=torch.float32, device=self.device)
        print('nodes {}, edes {}, features {}, classes {}'.format(self.num_nodes, self.num_edges, self.num_feas, self.num_classes))


    def _load_data(self):
        lines = linecache.getlines(self.label_file)
        # print(lines)
        lines = [line.rstrip('\n') for line in lines]
        # print(lines)

        #===========load label============
        node_map = {}
        label_map = {}
        Y = []
        cnt = 0
        for idx, line in enumerate(lines):
            line = line.split(' ')
            node_map[line[0]] = idx
            y = []
            for label in line[1:]:
                if label not in label_map:
                    label_map[label] = cnt
                    cnt += 1
                y.append(label_map[label])
            Y.append(y)
        num_classes = len(label_map)
        num_nodes = len(node_map)

        L = np.array([la[0] for la in Y])



        #=========load feature==========
        lines = linecache.getlines(self.feature_file)
        lines = [line.rstrip('\n') for line in lines]
        # print('line:', lines)

        num_features = len(lines[0].split(' ')) - 1
        X = np.zeros((num_nodes, num_features), dtype=np.float32)

        for line in lines:
            line = line.split(' ')
            node_id = node_map[line[0]]
            X[node_id] = np.array([float(x) for x in line[1:]])


        #==========load graph========
        A = np.zeros((num_nodes, num_nodes))
        lines = linecache.getlines(self.graph_file)
        lines = [line.rstrip('\n') for line in lines]
        for line in lines:
            line = line.split(' ')
            idx1 = node_map[line[0]]
            idx2 = node_map[line[1]]
            A[idx2, idx1] = 1.0
            A[idx1, idx2] = 1.0



        #=========load walks========
        W = np.zeros((num_nodes, num_nodes))
        lines = linecache.getlines(self.walks_file)
        lines = [line.rstrip('\n') for line in lines]
        for line in lines:
            line = line.split(' ')
            idx1 = node_map[line[0]]
            idx2 = node_map[line[1]]
            W[idx2, idx1] = 1.0
            W[idx1, idx2] = 1.0

        return A, X, W, L, num_classes



