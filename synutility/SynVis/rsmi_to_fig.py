import networkx as nx
import matplotlib.pyplot as plt
from typing import Union, Tuple


from synutility.SynVis.graph_visualizer import GraphVisualizer
from synutility.SynIO.Format.smi_to_graph import rsmi_to_graph
from synutility.SynIO.Format.its_construction import ITSConstruction

vis_graph = GraphVisualizer()


def three_graph_vis(
    input: Union[str, Tuple[nx.Graph, nx.Graph, nx.Graph]],
    sanitize: bool = False,
    figsize: Tuple[int, int] = (18, 5),
    orientation: str = "horizontal",
    show_titles: bool = True,
    show_atom_map: bool = False,
) -> plt.Figure:
    """
    Visualize three related graphs (reactants, intermediate transition state, and products)
    side by side or vertically in a single figure.

    Parameters:
    - input (Union[str, Tuple[nx.Graph, nx.Graph, nx.Graph]]): Either a reaction SMILES string
      or a tuple of three NetworkX graphs (reactants, ITS, products).
    - sanitize (bool, optional): If True, sanitizes the input molecule. Default is False.
    - figsize (Tuple[int, int], optional): The size of the Matplotlib figure. Default is (18, 5).
    - orientation (str, optional): Layout of the subplots; 'horizontal' or 'vertical'. Default is 'horizontal'.
    - show_titles (bool, optional): If True, adds titles to each subplot. Default is True.

    Returns:
    - plt.Figure: The Matplotlib figure containing the three subplots.
    """
    try:
        # Parse input to determine graphs
        if isinstance(input, str):
            r, p = rsmi_to_graph(input, light_weight=True, sanitize=sanitize)
            its = ITSConstruction().ITSGraph(r, p)
        elif isinstance(input, tuple) and len(input) == 3:
            r, p, its = input
        else:
            raise ValueError(
                "Input must be a reaction SMILES string or a tuple of three graphs (r, p, its)."
            )

        # Set up subplots
        if orientation == "horizontal":
            fig, ax = plt.subplots(1, 3, figsize=figsize)
        elif orientation == "vertical":
            fig, ax = plt.subplots(3, 1, figsize=figsize)
        else:
            raise ValueError("Orientation must be 'horizontal' or 'vertical'.")

        # Plot the graphs
        vis_graph.plot_as_mol(
            r,
            ax[0],
            show_atom_map=show_atom_map,
            font_size=12,
            node_size=800,
            edge_width=2.0,
        )
        if show_titles:
            ax[0].set_title("Reactants")

        vis_graph.plot_its(its, ax[1], use_edge_color=True, show_atom_map=show_atom_map)
        if show_titles:
            ax[1].set_title("Imaginary Transition State")

        vis_graph.plot_as_mol(
            p,
            ax[2],
            show_atom_map=show_atom_map,
            font_size=12,
            node_size=800,
            edge_width=2.0,
        )
        if show_titles:
            ax[2].set_title("Products")

        return fig

    except Exception as e:
        raise RuntimeError(f"An error occurred during visualization: {str(e)}")
