import networkx as nx
import pandas as pd
from ogb.linkproppred import LinkPropPredDataset

print('Loading dataset...')
dataset = LinkPropPredDataset(name='ogbl-ppa')

split_edge = dataset.get_edge_split()
train_edge, valid_edge, test_edge = split_edge["train"], split_edge["valid"], split_edge["test"]
graph = dataset[0]  # graph: library-agnostic graph object

G = nx.Graph()

print('Creating graph...')
for i in range(graph['num_nodes']):
    source = graph['edge_index'][0][i].item()
    target = graph['edge_index'][1][i].item()
    G.add_edge(source, target)

print('Calculating degree per node...')
degrees = dict(G.degree())

# Save degree per node in pd.DataFrame
degrees_df = pd.DataFrame(degrees.items(), columns=['Node', 'Degree'])

# Save degree per node in CSV
degrees_df.to_csv('../results/degree_per_node.csv', index=False)

print('Done!')
