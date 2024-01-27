from torch_geometric.utils import to_networkx
from torch_geometric.data import Data

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
