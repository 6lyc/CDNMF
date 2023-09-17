from Utils.walks import *

if __name__=='__main__':
    graph_config = {
        'is_adjlist': False,
        'graph_file': './pubmed/edges.txt',
        'label_file': './pubmed/labels.txt',
        'feature_file': './pubmed/features.txt',
        'node_status_file': '',
    }

    walk_config = {
        'num_walks': 10,
        'walk_length': 40,
        'window_size': 10,
        'walks_file': './pubmed/walks.txt'
    }

    graph = Graph(graph_config)
    get_walks(graph, walk_config)

