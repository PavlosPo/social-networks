# Better than louvain
import cugraph as cg
import cudf
import networkx as nx
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

print('Calculating Leiden Communities...')
# Calculate Leiden Communities
parts, modularity_score = cg.community.leiden(G)

# Save communities to CSV
df = pd.DataFrame([(node_id, partition_id) for node_id, partition_id in parts.items()], columns=['NodeID', 'PartitionID'])
df.to_csv('leiden_communities.csv', index=False)