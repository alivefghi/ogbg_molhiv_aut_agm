from ogb.graphproppred.dataset_pyg import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from Utils.utils import pyg_to_networkx, get_n_from_each_group
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