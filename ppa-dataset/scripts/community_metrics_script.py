
import networkx as nx
import pandas as pd
from ogb.linkproppred import LinkPropPredDataset
from networkx.algorithms import community, cuts

print('Loading dataset...')
dataset = LinkPropPredDataset(name='ogbl-ppa')

split_edge = dataset.get_edge_split()
train_edge, valid_edge, test_edge = split_edge["train"], split_edge["valid"], split_edge["test"]
graph = dataset[0] # graph: library-agnostic graph object

print('Creating graph...')

# Initialize CuGraph graph directly from edge list

G = nx.Graph()

for i in range(graph['num_nodes']):
    source = graph['edge_index'][0][i].item()
    target = graph['edge_index'][1][i].item()
    G.add_edge(source, target)

def calculate_metrics(G, communities):
    modularity = nx.community.modularity(G, communities)
    conductance_vals = []
    for i, community in enumerate(communities):
        print('Trying next community of size: ', len(community))
        print('Remaining communities: ', len(communities) - i - 1)
        other_nodes = set(G.nodes) - set(community)
        conductance_vals.append(cuts.conductance(G, community, other_nodes))
    avg_conductance = sum(conductance_vals) / len(conductance_vals)
    coverage, performance = nx.algorithms.community.partition_quality(G, communities)

    return modularity, avg_conductance, coverage, performance

# Leiden
df = pd.read_csv("../../results/leiden_communities.csv")
list_with_nodes_in_communities = df.groupby('PartitionID')['Node'].apply(list).reset_index(name='Nodes')
leiden_communities = list_with_nodes_in_communities['Nodes']
leiden_metrics = calculate_metrics(G, leiden_communities)

# Louvain
df = pd.read_csv("../../results/louvain_communities_format_2.csv")
list_with_nodes_in_communities = df.groupby('CommunityID')['Node'].apply(list).reset_index(name='Nodes')
louvain_communities = list_with_nodes_in_communities['Nodes']
louvain_metrics = calculate_metrics(G, louvain_communities)

print(f'Leiden Metrics: Modularity={leiden_metrics[0]}, Avg. Conductance={leiden_metrics[1]}, Coverage={leiden_metrics[2]}, Performance={leiden_metrics[3]}')
print(f'Louvain Metrics: Modularity={louvain_metrics[0]}, Avg. Conductance={louvain_metrics[1]}, Coverage={louvain_metrics[2]}, Performance={louvain_metrics[3]}')

# Save results in dataframe
data = {'Algorithm': ['Leiden', 'Louvain'],
        'Modularity': [leiden_metrics[0], louvain_metrics[0]],
        'Avg. Conductance': [leiden_metrics[1], louvain_metrics[1]],
        'Coverage': [leiden_metrics[2], louvain_metrics[2]],
        'Performance': [leiden_metrics[3], louvain_metrics[3]]}
df = pd.DataFrame(data)
df.to_csv("../../results/community_detection_metrics.csv", index=False)