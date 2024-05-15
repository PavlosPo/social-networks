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
# Extract edge data
# edges_df = pd.DataFrame({'source': graph['edge_index'][0], 'target': graph['edge_index'][1]})
# edges_cudf = cudf.from_pandas(edges_df)

# Initialize CuGraph graph directly from edge list
G = nx.Graph()

for i in range(graph['num_nodes']):
    source = graph['edge_index'][0][i]
    target = graph['edge_index'][1][i]
    G.add_edge(source, target)

print('Calculating degree centrality...')
# Calculate katz centrality
cc_scores = cg.degree_centrality(G)

# Save in pd.DataFrame
print('Saving results...')
results_df = pd.DataFrame(cc_scores.items(), columns=['Node', 'Degree Centrality'])

# Save results in csv
results_df.to_csv('degree_centrality.csv', index=False)

print('Done!')
