"""This module contains helpder functions to visualise directed acyclic graphs
"""
from math import ceil
import matplotlib.pyplot as plt
from daft import PGM
from IPython.display import IFrame

from networkx import draw_networkx, DiGraph
from pgmpy.models import BayesianNetwork
from pyvis.network import Network

def __set_size(width : int ,height : int):
    """Explicitly sets the size of a matplotlib plot

    Args:
        width (int): Width in inches
        height (int): Height in inches

    References:
        https://stackoverflow.com/questions/44970010/axes-class-set-explicitly-size-width-height-of-axes-in-given-units
    """
    axis=plt.gca()

    figw = float(width)/(axis.figure.subplotpars.right - axis.figure.subplotpars.left)
    figh = float(height)/(axis.figure.subplotpars.top-axis.figure.subplotpars.bottom)

    axis.figure.set_size_inches(figw, figh)

def display_daft_model(model : BayesianNetwork, figsize : tuple = (2, 2)) -> PGM:
    """Plots a small-format directed acyclic graph with black-and-white nodes that can be positioned

    Args:
        model (BayesianNetwork): The model to be drawn which must implement an edges collection
        pos (dict, optional): A dictionary with nodes as keys and positions as values. Each value is a list containing an X and Y co-ordinate. If not specified a spring layout positioning will be computed. Defaults to None (automatic layout).
        figsize (tuple, optional): Sets the figure size. If None default (small) sizing is used. Defaults to (2, 2).

    Example:
        >>> from pgmpy.models import BayesianNetwork

        >>> domain_model = BayesianNetwork([('Vaccination?', 'Reaction?'), ('Vaccination?', 'Smallpox?'), ('Reaction?', 'Death?'), ('Smallpox?', 'Death?')])
        >>> plot_daft_model(domain_model, pos=POS)
    """
    model.to_daft().render()
    __set_size(figsize[0], figsize[1])
    plt.show()

def display_networkx_model(model : BayesianNetwork, pos : dict = None, figsize : tuple = (10, 8), node_size : int = 5000, auto_layout_cols : int = 0):
    """Plots a large-format directed acyclic graph with coloured nodes that can be sized and positioned

    Args:
        model (BayesianNetwork): The model to be drawn which must implement an edges collection
        pos (dict, optional): A dictionary with nodes as keys and positions as values. Each value is a list containing an X and Y co-ordinate. If not specified a spring layout positioning will be computed. Defaults to None (automatic layout).
        figsize (tuple, optional): The size of the displayed plot. Defaults to (10, 8).
        node_size (int, optional): The size of the nodes in the plot. Defaults to 5000.
        auto_layout_cols (int, optional): If set > 0 an auto-layout is generated with the specified number of columns. Defaults to 0.

    Example:
        >>> from pgmpy.models import BayesianNetwork

        >>> POS : dict = {'Vaccination?': [0, 1], 'Reaction?': [-1, 0], 'Smallpox?': [1, 0], 'Death?': [0, -1]}
        >>> domain_model = BayesianNetwork([('Vaccination?', 'Reaction?'), ('Vaccination?', 'Smallpox?'), ('Reaction?', 'Death?'), ('Smallpox?', 'Death?')])
        >>> plot_model(model=domain_model, pos=POS)
    """
    if auto_layout_cols > 0:
        node_pos = [[x, y] for x in range(auto_layout_cols) for y in range(ceil(len(model.nodes)/auto_layout_cols))]
        pos = {k: node_pos[i] for i, k in enumerate(list(model.nodes))}

    plt.figure(figsize=figsize)
    plt.box(False)
    draw_networkx(DiGraph(model.edges), with_labels=True, pos=pos, node_color=[f"C{i}" for i in range(len(model.nodes))], node_size=node_size, arrowsize=25)
    plt.show()

def display_pyvis_model(model : BayesianNetwork, figsize : tuple = (500, 500), notebook : bool = True, enable_physics : bool = False, hierarchical : bool = False, filename : str = "pyvis.html") -> IFrame:
    """Displays a fully interactive directed acyclic graph that can either be embedded in a Jupyter Notebook cell or displayed in a new browser window

    Args:
        model (BayesianNetwork): The model to be drawn which must implement an edges collection
        figsize (tuple, optional): The size of the displayed plot. Defaults to (10, 8)
        notebook (bool, optional): If True the graph is displayed inside a Jupyter Notebook cell, if false it is launched in a separate browser window. Defaults to True
        enable_physics (bool, optional): If True the nodes "jiggle" when they are dragged which looks really neat but it can mean that it is difficult to get a good lauout. False turns the "jiggle" off. Defaults to False.
        filename (str, optional): The temporary filename used to save and store the HTML output. Defaults to "pyvis.html"

    Returns:
        IFrame: An IFrame is returned if notebook = True so that it can be embedded in the Jupyter Notebook cell. If notebook = False there is no return value as the graph is rendered in a separate browser window

    Examples:
        >>> from pgmpy.models import BayesianNetwork
        >>> domain_model = BayesianNetwork([('Vaccination?', 'Reaction?'), ('Vaccination?', 'Smallpox?'), ('Reaction?', 'Death?'), ('Smallpox?', 'Death?')])

        >>> # Display within the Jupyter Notebook cell
        >>> display_pyvis_model(model=model)

        >>> # Display in a new browser window (including all menus)
        >>> display_pyvis_model(model=model, notebook=False)

        >>> # Display within the Jupyter Notebook cell and make the default size larger by 50%
        >>> display_pyvis_model(model=model, figsize=(750, 750))
    """
    net = Network(height=f"{figsize[0]}px", width=f"{figsize[1]}px", notebook=notebook, directed=True, heading="") # Set up the Network object
    #if not notebook: # If displaying in a separate browser window ensure the buttons window is displayed
    #    net.show_buttons()

    # https://towardsdatascience.com/making-network-graphs-interactive-with-python-and-pyvis-b754c22c270
    # https://github.com/WestHealth/pyvis/issues/48

    net.from_nx(DiGraph(model.edges)) # Build the nodes and edges directly from the BayesianNetwork

    if notebook:
        options = 'var options = {"edges": {"color": {"inherit": true},"smooth": false}'
        if not enable_physics:
            options = options + ',"physics": {"enabled": false,"minVelocity": 0.75}'
        if hierarchical:
            options = options + ',"layout": {"hierarchical": {"enabled": true}}'
        options = options + "}"

        # if not enable_physics:
        #     options = r'var options = {"edges": {"color": {"inherit": true},"smooth": false},"physics": {"enabled": false,"minVelocity": 0.75}}'

        #     net.set_options(options)

        net.set_options(options)
    else:
        net.show_buttons()

    net.show(filename) # Create the temporary file that contains the HTML and CSS

    if notebook: # If displaying and embedding in a Jupyter Notebook cell an IFrame must be returned or nothing will be rendered
        return IFrame(src=filename, width=figsize[0]*1.1, height=figsize[1]*1.1)

# def cpd_to_dataframe(tabularcpd : TabularCPD) -> pd.DataFrame:
#     """Converts a TabularCPD (pgmpy Conditional Probability Table object) output into a DataFrame

#     Args:
#         tabularcpd (tabularcpd): The TabularCPD to be converted.

#     Returns:
#         pd.DataFrame: The output DataFrame

#     Notes:
#         Modified from _make_table_str in CPD.py (right click on TabularCPD and choose Goto Definition)
#     """
#     headers_list = []

#     # Build column headers
#     evidence = tabularcpd.variables[1:]
#     evidence_card = tabularcpd.cardinality[1:]
#     if evidence:
#         col_indexes = np.array(list(product(*[range(i) for i in evidence_card])))
#         if tabularcpd.state_names:
#             for i in range(len(evidence_card)):
#                 column_header = [str(evidence[i])] + [
#                     "{var}({state})".format(
#                         var=evidence[i], state=tabularcpd.state_names[evidence[i]][d]
#                     )
#                     for d in col_indexes.T[i]
#                 ]
#                 headers_list.append(column_header)
#         else:
#             for i in range(len(evidence_card)):
#                 column_header = [str(evidence[i])] + [
#                     f"{evidence[i]}_{d}" for d in col_indexes.T[i]
#                 ]
#                 headers_list.append(column_header)

#     # Build row headers
#     if tabularcpd.state_names:
#         variable_array = [
#             [
#                 "{var}({state})".format(
#                     var=tabularcpd.variable, state=tabularcpd.state_names[tabularcpd.variable][i]
#                 )
#                 for i in range(tabularcpd.variable_card)
#             ]
#         ]
#     else:
#         variable_array = [
#             [f"{tabularcpd.variable}_{i}" for i in range(tabularcpd.variable_card)]
#         ]
#     # Stack with data
#     labeled_rows = np.hstack(
#         (np.array(variable_array).T, tabularcpd.get_values())
#     ).tolist()

#     # If there is no headers list (because there is only one variable) then create one ready for the DataFrame conversion
#     if headers_list == []:
#         headers_list = [[tabularcpd.variables[0], "P"]]

#     # Convert the data and headers arrays into a DataFrame
#     df_cpd = pd.DataFrame(data=labeled_rows, columns=headers_list)

#     # If there is only one variable this is a "header node" i.e. one with no inputs / parents so the default format provided looks best
#     # If there are multiple variables i.e. a node with one or more inputs / parents then transposing the DataFrame give an output format
#     # that closely resembles that proposed by Pearl in "The Book of Why", Chapter 3, p117
#     if len(tabularcpd.variables) > 1:
#         df_cpd = df_cpd.T
#         df_cpd.columns = df_cpd.iloc[0]
#         df_cpd = df_cpd.iloc[1: , :]

#     df_cpd = df_cpd.apply(pd.to_numeric, errors="ignore")
#     return df_cpd

# def display_cpd(tabularcpd : TabularCPD):
#     """Displays a CPD to a Jupyter cell output including a formatted explanation and a DataFram of the probabilities

#     Args:
#         tabularcpd (tabularcpd): The TabularCPD to be displayed.
#     """
#     description = f"Probability of {tabularcpd.variables[0]}"
#     if len(tabularcpd.variables) > 1:
#         description = description + f" given {', '.join(tabularcpd.variables[1:])}"

#     df_cpd = cpd_to_dataframe(tabularcpd)

#     print(description)
#     display(df_cpd)

# def display_query(query : DiscreteFactor, phi_or_p : str = "phi", print_state_names : bool = True) -> pd.DataFrame:
#     """Converts a DiscreteFactor (a pgmpy query object) output into a DataFrame

#     Args:
#         phi_or_p (str, optional): 'phi': When used for Factors. 'p': When used for CPDs. Defaults to "phi".
#         print_state_names (bool, optional): If True, the user defined state names are displayed.. Defaults to True.

#     Returns:
#         pd.DataFrame: The output DataFrame

#     Notes:
#         Modified from _str in DiscreteFactor.py (right click on TabularCPD and choose Goto Definition)
#     """
#     string_header = list(map(str, query.scope()))
#     string_header.append(f"{phi_or_p}({','.join(string_header)})")

#     value_index = 0
#     factor_table = []
#     for prob in product(*[range(card) for card in query.cardinality]):
#         if query.state_names and print_state_names:
#             prob_list = [
#                 "{var}({state})".format(
#                     var=list(query.variables)[i],
#                     state=query.state_names[list(query.variables)[i]][prob[i]],
#                 )
#                 for i in range(len(query.variables))
#             ]
#         else:
#             prob_list = [
#                 f"{list(query.variables)[i]}_{prob[i]}"
#                 for i in range(len(query.variables))
#             ]

#         prob_list.append(query.values.ravel()[value_index])
#         factor_table.append(prob_list)
#         value_index += 1

#     return pd.DataFrame(factor_table, columns=string_header)
