from Utils.walks import *

if __name__=='__main__':
    graph_config = {
        'is_adjlist': False,
        'graph_file': './citeseer/edges.txt',
        'label_file': './citeseer/labels.txt',
        'feature_file': './citeseer/features.txt',
        'node_status_file': '',
    }

    walk_config = {
        'num_walks': 10,
        'walk_length': 80,
        'window_size': 10,
        'walks_file': './citeseer/walks.txt'
    }

    graph = Graph(graph_config)
    get_walks(graph, walk_config)

