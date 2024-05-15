import networkx as nx
import pandas as pd
from ogb.linkproppred import LinkPropPredDataset

print('Loading dataset...')
dataset = LinkPropPredDataset(name = 'ogbl-ppa')

split_edge = dataset.get_edge_split()
train_edge, valid_edge, test_edge = split_edge["train"], split_edge["valid"], split_edge["test"]
graph = dataset[0] # graph: library-agnostic graph object

G = nx.Graph()

print('Creating graph...')
for i in range(graph['num_nodes']):
    source = graph['edge_index'][0][i]
    target = graph['edge_index'][1][i]
    G.add_edge(source, target)


print('Calculating harmonic centrality...')
# # calculate closeness
results = nx.closeness_centrality(G)

# Save in pd.DataFrame
print('Saving results...')
results_df = pd.DataFrame(results.items(), columns=['Node', 'Closeness Centrality'])

# Save results in csv
results_df.to_csv('closeness_centrality.csv', index=False)

print('Done!')