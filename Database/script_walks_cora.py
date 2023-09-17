import sys
sys.path.append("..")
from Utils import walks

if __name__=='__main__':
    graph_config = {
        'is_adjlist': False,
        'graph_file': './cora/edges.txt',
        'label_file': './cora/labels.txt',
        'feature_file': './cora/features.txt',
        'node_status_file': '',
    }

    walk_config = {
        'num_walks': 10,
        'walk_length': 80,
        'window_size': 10,
        'walks_file': './cora/walks.txt'
    }

    graph = walks.Graph(graph_config)
    walks.get_walks(graph, walk_config)

