from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from ogb.graphproppred.dataset_pyg import PygGraphPropPredDataset

import networkx as nx


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