from random import random
import networkx as nx
import itertools
import pandas as pd
from ogb.linkproppred import LinkPropPredDataset


def most_central_edge(G):
    # Calculate PageRank centrality for each edge
    pr = nx.pagerank(G)
    
    # Find the edge with the maximum PageRank centrality
    max_pr = max(pr.values())
    pr = {e: c / max_pr for e, c in pr.items()}  # Scale PageRank values between 0 and 1
    pr = {e: c + random() for e, c in pr.items()}  # Add some random noise
    
    return max(pr, key=pr.get)

print('Loading dataset...')
dataset = LinkPropPredDataset(name='ogbl-ppa')

split_edge = dataset.get_edge_split()
train_edge, valid_edge, test_edge = split_edge["train"], split_edge["valid"], split_edge["test"]
graph = dataset[0]  # graph: library-agnostic graph object

print('Creating graph...')
# Initialize CuGraph graph directly from edge list
G = nx.Graph()

for i in range(graph['num_nodes']):
    source = graph['edge_index'][0][i]
    target = graph['edge_index'][1][i]
    G.add_edge(source, target)


k = 58 # 58 Communities 
comp = nx.community.girvan_newman(G, most_valuable_edge=most_central_edge)

limited = itertools.takewhile(lambda c: len(c) <= k, comp)
for communities in limited:
    all_communities = tuple(sorted(c) for c in communities) # Last loop gets the k communities.

nodes_specified_to_community_id = { node:i for i, c in enumerate(all_communities) for node in c}

# Save communities to CSV
df = pd.DataFrame([(node_id, partition_id) for node_id, partition_id in nodes_specified_to_community_id.items()], columns=['NodeID', 'PartitionID'])
df.to_csv('girvan__pagerank_communities.csv', index=False)
