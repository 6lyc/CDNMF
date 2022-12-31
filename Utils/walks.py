import networkx as nx
import numpy as np
import linecache
import random
from scipy.sparse import dok_matrix



class Graph(object):
    def __init__(self, config):
        self.G = None
        self.is_adjlist = config['is_adjlist']
        self.graph_file = config['graph_file']
        self.label_file = config['label_file']
        self.feature_file = config['feature_file']
        self.node_status_file = config['node_status_file']

        if self.is_adjlist:
            self.read_adjlist()
        else:
            self.read_edgelist()


        if self.label_file:
            self.read_node_label()

        if self.feature_file:
            self.read_node_features()

        if self.node_status_file:
            self.read_node_status()


        self.num_nodes = self.G.number_of_nodes()
        self.num_edges = self.G.number_of_edges()

        print('num of nodes: {}'.format(self.num_nodes))
        print('num of edges: {}'.format(self.num_edges))


    def encode_node(self):
        for id, node in enumerate(self.G.nodes()):
            self.G.nodes[node]['id'] = id
            self.G.nodes[node]['status'] = ''



    def read_adjlist(self):
        self.G = nx.read_adjlist(self.graph_file, create_using=nx.DiGraph())
        for i, j in self.G.edges():
            self.G[i][j]['weight'] = 1.0

        self.encode_node()

    def read_edgelist(self):
        self.G = nx.DiGraph()

        lines = linecache.getlines(self.graph_file)
        lines = [line.rstrip('\n') for line in lines]

        for line in lines:
            line = line.split(' ')

            src = line[0]
            dst = line[1]

            self.G.add_edge(src, dst)
            self.G.add_edge(dst, src)

            weight = 1.0
            if len(line) == 3:
                weight = float(line[2])
            self.G[src][dst]['weight'] = float(weight)
            self.G[dst][src]['weight'] = float(weight)

        self.encode_node()

    def read_node_label(self):
        lines = linecache.getlines(self.label_file)
        lines = [line.rstrip('\n') for line in lines]

        for line in lines:
            line = line.split(' ')
            self.G.nodes[line[0]]['label'] = line[1:]

    def read_node_features(self):
        lines = linecache.getlines(self.feature_file)
        lines = [line.rstrip('\n') for line in lines]

        for line in lines:
            line = line.split(' ')
            self.G.nodes[line[0]]['feature'] = np.array([float(x) for x in line[1:]])

    def read_node_status(self):
        lines = linecache.getlines(self.feature_file)
        lines = [line.rstrip('\n') for line in lines]

        for line in lines:
            line = line.split(' ')
            self.G.nodes[line[0]]['status'] = line[1] # train test valid


class DeepWalker:
    def __init__(self, G):
        self.G = G.G


    def deepwalk_walk(self, walk_length, start_node):
        G = self.G
        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break
        return walk

    def simulate_walks(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print('Walk iteration:')
        for walk_iter in range(num_walks):
            print(str(walk_iter+1) + '/' + str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.deepwalk_walk(walk_length=walk_length, start_node=node))
        return walks


def get_walks(graph, config):
    num_walks = config['num_walks']
    walk_length = config['walk_length']
    window_size = config['window_size']
    walks_file = config['walks_file']

    walker = DeepWalker(graph)
    walks = walker.simulate_walks(num_walks, walk_length)

    num_nodes = graph.num_nodes
    adj_matrix = dok_matrix((num_nodes, num_nodes), np.float32)

    node_map = {} #id->node
    for node in graph.G.nodes():
        node_map[graph.G.nodes[node]['id']] = node

    for line in walks:
        for pos, node in enumerate(line):
            start = max(0, pos - window_size)
            for pos2, node2 in enumerate(line[start:(pos + window_size + 1)], start):
                if pos2 != pos:
                    src = graph.G.nodes[node]['id'] #node->id
                    dst = graph.G.nodes[node2]['id']
                    adj_matrix[src, dst] = 1.0
                    adj_matrix[dst, src] = 1.0

    edge_list = []
    for item in adj_matrix.items():
        src = item[0][0]
        dst = item[0][1]
        if dst > src:
            edge_list.append(node_map[src] + ' ' + node_map[dst])  #id->node

    with open(walks_file, 'w') as fid:
        fid.write('\n'.join(edge_list))





