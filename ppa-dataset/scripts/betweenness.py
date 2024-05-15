# %load_ext cudf.pandas
# pandas API is now GPU accelerated

import pandas as pd
import networkx as nx
import cugraph as cg
import cudf
from cugraph.experimental import PropertyGraph
from ogb.linkproppred import LinkPropPredDataset
dataset = LinkPropPredDataset(name='ogbl-ppa')

split_edge = dataset.get_edge_split()
train_edge, valid_edge, test_edge = split_edge["train"], split_edge["valid"], split_edge["test"]
graph = dataset[0]  # graph: library-agnostic graph object

G = nx.Graph()

for i in range(graph['num_nodes']):
    source = graph['edge_index'][0][i]
    target = graph['edge_index'][1][i]
    G.add_edge(source, target)
    
# Calculate betweenness centrality
bc_scores = cg.betweenness_centrality(G)

# Save in pd.DataFrame
results_df = pd.DataFrame(bc_scores.items(), columns=['Node', 'Betweenness Centrality'])

# Save it in csv
results_df.to_csv('betweenness_centrality.csv', index=False)
