# Better than louvain
import networkx as nx
import itertools
import pandas as pd
from ogb.linkproppred import LinkPropPredDataset

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

print('Calculating Girvan Communities...')
# Calculate Leiden Communities

k = 58
comp = nx.community.girvan_newman(G)

limited = itertools.takewhile(lambda c: len(c) <= k, comp)
for communities in limited:
    all_communities = tuple(sorted(c) for c in communities) # Last loop gets the k communities.

nodes_specified_to_community_id = { node:i for i, c in enumerate(all_communities) for node in c}

# Save communities to CSV
df = pd.DataFrame([(node_id, partition_id) for node_id, partition_id in nodes_specified_to_community_id.items()], columns=['NodeID', 'PartitionID'])
df.to_csv('girvan_communities.csv', index=False)