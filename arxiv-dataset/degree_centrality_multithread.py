import networkx as nx
import multiprocessing
from ogb.nodeproppred import NodePropPredDataset
import random

def calculate_degree_centrality(G):
    return nx.degree_centrality(G)

if __name__ == '__main__':
    # Load your graph G here
    print('Script Begins...')
    # Create a graph (example)
    G = nx.Graph()

    # Load the arXiv dataset
    dataset = NodePropPredDataset(name='ogbn-arxiv')

    graph = dataset[0][0]
    edge_index = graph['edge_index']  # edge index

    # Create a new directed graph
    G = nx.DiGraph()

    # Add edges to the graph
    for i in range(edge_index.shape[1]):
        source = edge_index[0][i]
        target = edge_index[1][i]
        G.add_edge(source, target)

    print('Parallelization Begins...')
    # Create a pool of worker processes
    with multiprocessing.Pool() as pool:
        # Parallel computation of degree centrality
        degree_centrality = pool.apply(calculate_degree_centrality, (G,))

    print('Outputs file named "degree_centrality_results.txt"')
    # Save degree centrality results to a text file
    output_filename = 'degree_centrality_results.txt'
    with open(output_filename, 'w') as f:
        for node, centrality in degree_centrality.items():
            f.write(f"Node: {node}, Centrality: {centrality}\n")

    print(f"Degree centrality results saved to {output_filename}")
