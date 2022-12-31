class Dataset(object):

    def __init__(self, config):

        self.num_nodes = self.A.shape[0]
        self.num_feas = self.X.shape[1]
        self.num_edges = np.sum(self.A) / 2

        self.A = torch.tensor(self.A, dtype=torch.float32, device=self.device)
        self.X = torch.tensor(self.X, dtype=torch.float32, device=self.device)
        self.W = torch.tensor(self.W, dtype=torch.float32, device=self.device)
        self.L = torch.tensor(self.L, dtype=torch.float32, device=self.device)
        print('nodes {}, edes {}, features {}, classes {}'.format(self.num_nodes, self.num_edges, self.num_feas, self.num_classes))