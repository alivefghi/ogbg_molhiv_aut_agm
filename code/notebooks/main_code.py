from ogb.graphproppred.dataset_pyg import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from Utils.utils import pyg_to_networkx, get_n_from_each_group, plot_networkx_graph
from tqdm import tqdm

import os
import torch
import networkx as nx
import matplotlib.pyplot as plt

os.environ["TORCH"] = torch.__version__
print(torch.__version__)

# Download and process data at './dataset/ogbg_molhiv/'
dataset = PygGraphPropPredDataset(name="ogbg-molhiv")

print(f"len(dataset): {len(dataset)} <- number of graphs")
print(f"dataset.num_classes: {dataset.num_classes} <- number of labels for classes")
print(
    f"dataset.num_node_features: {dataset.num_node_features} <- number of features for each node of graphs"
)
print(
    f"dataset.num_edge_features: {dataset.num_edge_features} <- number of features for each edge of graphs"
)

graph_number = 0
graph: Data = dataset[graph_number]
print(graph)
print("\n")
print(f"graph.num_nodes: {graph.num_nodes}")
print(f"graph.num_edges: {graph.num_edges}")
print(f"graph.has_isolated_nodes: {graph.has_isolated_nodes()}")
print(f"graph.has_self_loops: {graph.has_self_loops()}")
print(f"graph.is_directed: {graph.is_directed()}")
print("\n")
print("x:\t\t", graph.x.shape)
print(graph.x)
print("\n")
print("y:\t\t", graph.y.shape)
print(graph.y)
print("\n")
print("edge_attr:\t\t", graph.edge_attr.shape)
print(graph.edge_attr)
print("\n")
print("edge_index:\t\t", graph.edge_index.shape)
print(graph.edge_index)
print("\n")

graph: Data = dataset[0]
networkx_graph = pyg_to_networkx(graph)
print(f"graph type: {type(networkx_graph)}")
print(f"graph node attributes: {networkx_graph.nodes(data=True)}")
print(f"graph attributes: {networkx_graph.graph['y'][0][0]}")

(hiv_positive_graphs, hiv_negative_graphs) = get_n_from_each_group(n=5, dataset=dataset)

print(len(hiv_positive_graphs))
print(len(hiv_negative_graphs))

plt.figure(figsize=(50, 30))
for graph_number, networkx_graph in enumerate(
        hiv_positive_graphs + hiv_negative_graphs
):
    plot_networkx_graph(networkx_graph, graph_number)
plt.show()


# Node Degree

def DEachGraph(Graphs: list[nx.Graph]) -> list:
    degree_list = []
    for graph in Graphs:
        degree = dict(nx.degree(graph))
        degree_list.append(degree)
    return degree_list


plt.figure(figsize=(50, 30))
for graph_number, networkx_graph in enumerate(
        hiv_positive_graphs + hiv_negative_graphs
):
    if graph_number <= 4:
        plot_networkx_graph(
            networkx_graph, graph_number, DEachGraph(hiv_positive_graphs)[graph_number]
        )
    else:
        plot_networkx_graph(
            networkx_graph,
            graph_number,
            DEachGraph(hiv_negative_graphs)[graph_number - 5],
        )
plt.show()


# Eigenvector centrality

def ECEachGraph(Graphs: list[nx.Graph]) -> list:
    ec_list = []
    for graph in Graphs:
        eigenvector_centrality = nx.eigenvector_centrality(graph)
        ec_list.append(
            {key: round(value, 2) for key, value in eigenvector_centrality.items()}
        )
    return ec_list


plt.figure(figsize=(50, 30))
for graph_number, networkx_graph in enumerate(
        hiv_positive_graphs + hiv_negative_graphs
):
    if graph_number <= 4:
        plot_networkx_graph(
            networkx_graph, graph_number, ECEachGraph(hiv_positive_graphs)[graph_number]
        )
    else:
        plot_networkx_graph(
            networkx_graph,
            graph_number,
            ECEachGraph(hiv_negative_graphs)[graph_number - 5],
        )
plt.show()


# Betweenness centrality

def BCEachGraph(Graphs: list[nx.Graph]) -> list:
    bc_list = []
    for graph in Graphs:
        betweenness_centrality = nx.betweenness_centrality(graph)
        bc_list.append(
            {key: round(value, 2) for key, value in betweenness_centrality.items()}
        )
    return bc_list


# %%
plt.figure(figsize=(50, 30))
for graph_number, networkx_graph in enumerate(
        hiv_positive_graphs + hiv_negative_graphs
):
    if graph_number <= 4:
        plot_networkx_graph(
            networkx_graph, graph_number, BCEachGraph(hiv_positive_graphs)[graph_number]
        )
    else:
        plot_networkx_graph(
            networkx_graph,
            graph_number,
            BCEachGraph(hiv_negative_graphs)[graph_number - 5],
        )
plt.show()


# Closeness centrality

def ClCEachGraph(Graphs: list[nx.Graph]) -> list:
    clc_list = []
    for graph in Graphs:
        closeness_centrality = nx.closeness_centrality(graph)
        clc_list.append(
            {key: round(value, 2) for key, value in closeness_centrality.items()}
        )
    return clc_list


plt.figure(figsize=(50, 30))
for graph_number, networkx_graph in enumerate(
        hiv_positive_graphs + hiv_negative_graphs
):
    if graph_number <= 4:
        plot_networkx_graph(
            networkx_graph,
            graph_number,
            ClCEachGraph(hiv_positive_graphs)[graph_number],
        )
    else:
        plot_networkx_graph(
            networkx_graph,
            graph_number,
            ClCEachGraph(hiv_negative_graphs)[graph_number - 5],
        )
plt.show()


# Clustering coefficient

def CCEachGraph(Graphs: list[nx.Graph]) -> list:
    cc_list = []
    for graph in Graphs:
        clustering_coefficient = nx.clustering(graph)
        cc_list.append(
            {key: round(value, 2) for key, value in clustering_coefficient.items()}
        )
    return cc_list


plt.figure(figsize=(50, 30))
for graph_number, networkx_graph in enumerate(
        hiv_positive_graphs + hiv_negative_graphs
):
    if graph_number <= 4:
        plot_networkx_graph(
            networkx_graph, graph_number, CCEachGraph(hiv_positive_graphs)[graph_number]
        )
    else:
        plot_networkx_graph(
            networkx_graph,
            graph_number,
            CCEachGraph(hiv_negative_graphs)[graph_number - 5],
        )
plt.show()


#######################################################################################################################
# Extract Link Features

# 1. Distance-based feaures

def compute_all_shortest_paths(graph: nx.Graph):
    """
    Compute the shortest paths between every pair of nodes in the given NetworkX graph.

    Parameters:
    graph (networkx.Graph): Input graph.

    Returns:
    dict: A dictionary containing shortest path lengths between all pairs of nodes.
    """
    shortest_paths = dict(nx.all_pairs_shortest_path_length(graph))
    return shortest_paths


# %%
def plot_networkx_graph_link_features(
        networkX_graph: nx.Graph, graph_number: int, node_labels: dict = {}
) -> None:
    """Plot one of ten networkx graphs in 2*5 subplot figure
    Parameters
    ----------
    networkX_graph: nx.Graph
        The networkx graph
    graph_number: int
        The index number of graph to be inserted in figure as subplot
    label: dict
    Returns
    ----------
    None
    """
    node_color = "lightsteelblue"
    hiv = ""
    if networkX_graph.graph["y"][0][0] != 0:
        node_color = "lightcoral"
        hiv = "HIV "

    plt.subplot(2, 5, graph_number + 1, frameon=False)
    plt.title(f"{hiv}Graph # {str(graph_number + 1)}")
    plt.axis("off")

    options = {
        "node_color": node_color,
        "node_size": 1500,
        "font_size": 20,
        "width": 5,
        # "edge_alpha": 0.5,
    }

    if node_labels != {}:
        options["labels"] = node_labels

    node_color = "lightsteelblue"
    edge_colors = nx.get_edge_attributes(networkX_graph, 'color').values()
    edge_labels = nx.get_edge_attributes(networkX_graph, 'weight').values()

    # edge_labels = {(u, v): d['weight'] for u, v, d in networkX_graph.edges(data=True) if d['weight'] > 1}

    nx.draw_kamada_kawai(networkX_graph, with_labels=True, edge_color=edge_labels, **options)
    # pos = nx.kamada_kawai_layout(networkX_graph)
    # nx.draw(networkX_graph, pos, with_labels=True, edge_color=edge_colors, **options)
    # nx.draw_networkx_edge_labels(networkX_graph, pos, edge_labels=edge_labels)

    return None


# %%
def add_LF_as_new_edge(graph: nx.Graph, shortest_paths: dict, edge_color: str):
    """
    Add some edges to the graph considering the shortest path link feature
    """
    for source_node, paths in shortest_paths.items():
        for target_node, shortest_path_length in paths.items():
            if shortest_path_length < 2:
                continue
            link_weight = shortest_path_length
            graph.add_edge(source_node, target_node, weight=link_weight, color=edge_color)
    return graph


# %%
plt.figure(figsize=(50, 30))
for graph_number, networkx_graph in enumerate(
        hiv_positive_graphs + hiv_negative_graphs
):
    # Compute the shortest paths between every pair of nodes
    shortest_paths = compute_all_shortest_paths(networkx_graph)

    # Set blue color for all default graph edges
    edge_color = 'black'

    # Set 'color' attribute for all edges in the graph
    nx.set_edge_attributes(networkx_graph, edge_color, 'color')
    nx.set_edge_attributes(networkx_graph, 1, 'weight')
    new_graph = add_LF_as_new_edge(networkx_graph, shortest_paths, edge_color='red')

    plot_networkx_graph_link_features(
        new_graph, graph_number
    )
plt.show()


# Common Neighbors

def compute_all_common_neighbors(graph: nx.Graph):
    """
    Compute the shortest paths between every pair of nodes in the given NetworkX graph.

    Parameters:
    graph (networkx.Graph): Input graph.

    Returns:
    dict: A dictionary containing shortest path lengths between all pairs of nodes.
    """

    # Compute common neighbors as edge weights
    for u, v in graph.edges():
        common_neighbors = len(list(nx.common_neighbors(graph, u, v)))
        graph[u][v]['weight'] = common_neighbors

    return graph


# %%


for u, v in hiv_negative_graphs[0].edges():
    common_neighbors = len(list(nx.common_neighbors(hiv_negative_graphs[0], u, v)))
    hiv_negative_graphs[0][u][v]['weight'] = common_neighbors

nx.get_edge_attributes(hiv_negative_graphs[0], 'weight').values()

# %%
common_neighbors_feature_graph = compute_all_common_neighbors(hiv_negative_graphs[0])

print(common_neighbors_feature_graph.nodes())
print(common_neighbors_feature_graph.edges())
edge_colors = nx.get_edge_attributes(common_neighbors_feature_graph, 'color').values()
print(edge_colors)
edge_labels = nx.get_edge_attributes(common_neighbors_feature_graph, 'weight').values()
print(edge_labels)

# %%
# plt.figure(figsize=(50, 30))
for graph_number, networkx_graph in enumerate(
        hiv_positive_graphs + hiv_negative_graphs
):
    common_neighbors_feature_graph = compute_all_common_neighbors(networkx_graph)

    # # Set blue color for all default graph edges
    # edge_color = 'black'

    # # Set 'color' attribute for all edges in the graph
    # nx.set_edge_attributes(networkx_graph, edge_color, 'color')
    # nx.set_edge_attributes(networkx_graph, 1, 'weight')
    # new_graph = add_LF_as_new_edge(networkx_graph, shortest_paths, edge_color='red')

    # plot_networkx_graph_link_features(
    #         new_graph, graph_number
    #     )
# plt.show()
# %%
import networkx as nx

# Create a graph (replace this with your graph creation logic)
G = nx.Graph()
G.add_nodes_from([1, 2, 3, 4])
G.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4)])

# Compute common neighbors as edge weights
for u, v in G.edges():
    common_neighbors = len(list(nx.common_neighbors(G, u, v)))
    G[u][v]['weight'] = common_neighbors

# Print edge weights
for u, v, data in G.edges(data=True):
    print(f"Edge ({u}, {v}) - Common Neighbors: {data['weight']}")
# %%
nx.get_edge_attributes(G, 'weight').values()
# %%
import networkx as nx

# Create a graph (replace this with your graph creation logic)
G = nx.Graph()
G.add_nodes_from([1, 2, 3, 4])
G.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4)])

# Nodes for which you want to find common neighbors
node_u = 1
node_v = 2

# Compute common neighbors of nodes u and v
common_neighbors = list(nx.common_neighbors(G, node_u, node_v))

# Print the common neighbors
print(f"Common neighbors of nodes {node_u} and {node_v}: {common_neighbors}")

# %%
import networkx as nx
import matplotlib.pyplot as plt

# Create a graph (replace this with your graph creation logic)
G = nx.Graph()
G.add_nodes_from([1, 2, 3, 4])
G.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4)])

# Nodes for which you want to find common neighbors
node_u = 1
node_v = 2

# Compute common neighbors of nodes u and v
common_neighbors = list(nx.common_neighbors(G, node_u, node_v))

# Print the common neighbors
print(f"Common neighbors of nodes {node_u} and {node_v}: {common_neighbors}")

# Plot the graph
pos = nx.spring_layout(G)  # Position nodes using the spring layout algorithm
nx.draw(G, pos, with_labels=True, node_size=700, font_size=12, font_weight='bold', width=2)

# Highlight nodes u and v and their common neighbors
highlight_nodes = [node_u, node_v] + common_neighbors
node_colors = ['red' if node in highlight_nodes else 'skyblue' for node in G.nodes()]
nx.draw_networkx_nodes(G, pos, nodelist=highlight_nodes, node_color='red', node_size=700)
nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
nx.draw_networkx_edges(G, pos, width=2, edge_color='gray', alpha=0.7)

plt.title("Graph with Common Neighbors Highlighted")
plt.show()
