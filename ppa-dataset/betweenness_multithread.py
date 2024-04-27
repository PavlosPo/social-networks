import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ogb.linkproppred import LinkPropPredDataset
import pandas as pd
import numpy as np
import random
import multiprocessing

def calculate_betweenness_centrality(args):
    G, k = args
    if k is not None:
        nodes = random.sample(list(G.nodes()), min(k, len(G)))
        return nx.betweenness_centrality(G, k=k)
    else:
        return nx.betweenness_centrality(G)

if __name__ == '__main__':
    print('Script Began..')
    # Create a graph (example)
    G = nx.Graph()

    # Load the arXiv dataset
    dataset = LinkPropPredDataset(name = 'ogbl-ppa')

    graph = dataset[0]
    edge_index = graph['edge_index']  # edge index

    # Create a new directed graph
    G = nx.Graph()

    # Add edges to the graph
    for i in range(edge_index.shape[1]):
        source = edge_index[0][i]
        target = edge_index[1][i]
        G.add_edge(source, target)

    # Divide nodes into batches for parallel processing
    num_batches = 32
    node_batches = [list(G.nodes())[i::num_batches] for i in range(num_batches)]

    print('Parallelization Begins..')
    # Create a pool of worker processes
    with multiprocessing.Pool(processes=num_batches) as pool:
        # Parallel computation of betweenness centrality
        results = pool.map(calculate_betweenness_centrality, [(G, len(nodes)) for nodes in node_batches])


    # Merge results
    betweenness_centrality = {}
    for result in results:
        betweenness_centrality.update(result)

    print('Outputs file named "betweenness_centrality_results.txt"')
    # Save results to a text file
    output_filename = 'betweenness_centrality_results.txt'
    with open(output_filename, 'w') as f:
        for node, centrality in betweenness_centrality.items():
            f.write(f"Node: {node}, Centrality: {centrality}\n")

    print(f"Results saved to {output_filename}")
