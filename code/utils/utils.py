from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from ogb.graphproppred.dataset_pyg import PygGraphPropPredDataset

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import networkx as nx
from grakel import Graph


def pyg_to_networkx(pyg_graph: Data) -> nx.Graph:
    """Convert pyg graph to networkx graph

    Parameters
    ----------
    pyg_graph: torch_geometric.data.Data
        The pyg graph

    Returns
    ----------
    networkx_graph: nx.Graph
        The networkx graph
    """
    return to_networkx(
        pyg_graph,
        node_attrs=["x"],
        edge_attrs=["edge_attr"],
        graph_attrs=["y"],
        to_undirected=True,
    )




def networkx_to_grakel(networkx_graph: nx.Graph) -> Graph:
    # Create node labels
    node_labels = {node: node for node in networkx_graph.nodes()}

    # Convert to grakel.Graph
    return Graph(nx.to_dict_of_lists(networkx_graph), node_labels=node_labels, graph_format='dictionary')

def get_n_from_each_group(
        n: int, dataset: PygGraphPropPredDataset
) -> tuple[list[nx.Graph], list[nx.Graph]]:
    """Get n graphs of each binary grouped graphs from pyg dataset of graphs

    Parameters
    ----------
    dataset: PygGraphPropPredDataset
        The pyg graph dataset
    n: int
        The number of graph to be return in each binary group 0 and 1

    Returns
    ----------
    (hiv_positive_graphs, hiv_negative_graphs): tuple[list[nx.Graph], list[nx.Graph]]
        The tuple of positive hiv graphs and negative hiv graphs
    """
    enough_hiv_positive_graphs = n
    enough_hiv_negative_graphs = n
    hiv_positive_graphs: list[nx.Graph] = []
    hiv_negative_graphs: list[nx.Graph] = []
    for graph in dataset:
        if enough_hiv_negative_graphs == 0 and enough_hiv_positive_graphs == 0:
            break
        if (graph.y[0][0] == 0) and enough_hiv_negative_graphs > 0:
            networkx_graph = pyg_to_networkx(graph)
            hiv_negative_graphs.append(networkx_graph)
            enough_hiv_negative_graphs = enough_hiv_negative_graphs - 1
        if (graph.y[0][0] == 1) and enough_hiv_positive_graphs > 0:
            networkx_graph = pyg_to_networkx(graph)
            hiv_positive_graphs.append(networkx_graph)
            enough_hiv_positive_graphs = enough_hiv_positive_graphs - 1

    return hiv_positive_graphs, hiv_negative_graphs


def plot_networkx_graph(
        networkX_graph: nx.Graph, graph_number: int, labels: dict = {}
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
    }

    if labels != {}:
        options["labels"] = labels

    nx.draw_kamada_kawai(networkX_graph, with_labels=True, **options)

    return None
