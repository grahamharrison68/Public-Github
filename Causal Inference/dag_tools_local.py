from math import ceil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import IFrame, SVG, display

from networkx import draw_networkx, DiGraph
from pgmpy.models import BayesianNetwork
from pyvis.network import Network
from graphviz import Digraph
from pygraphviz import AGraph
import daft
from daft import PGM
from sklearn.preprocessing import LabelEncoder

def set_size(width : int ,height : int):
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

class DirectedAcyclicGraph():
    """Encapsulates all of the functionality of a causal network
    """

    def __init__(self, edges : list = None, digraph_string : str = ""):
        """Constructor

        Args:
            edges (list): A list of tuples representing the edges
            digraph_string (str, optional): If provided this is a string representing a graph in digraph format which will be used to build the edges collection. Defaults to "".

        Example:
            >>> climate_network = CausalNetwork([('POP', 'EC'),   ('URB', 'EC'),   ('GDP', 'EC'),
            >>>                                  ('EC', 'FFEC'),  ('EC', 'REC'),   ('EC', 'EI'),
            >>>                                  ('REC', 'CO2'),  ('REC', 'CH4'),  ('REC', 'N2O'),
            >>>                                  ('FFEC', 'CO2'), ('FFEC', 'CH4'), ('FFEC', 'N2O')])
        """
        if digraph_string != "":
            edges = AGraph(digraph_string).edges() # AGraph can ingest a DiGraph string and automatically convert it to a model with nodes and edges so no additional work is necessary

        self.__bayesian_network = BayesianNetwork(edges)

    @property
    def bayesian_network(self) -> BayesianNetwork:
        """Some causal inference libraries (i.e. pgmpy) require a BayesianNetwork instance to build the inferencing model, hence this property provides access

        Returns:
            BayesianNetwork: The underlying BayesianNetwork
        """
        return self.__bayesian_network

    @property
    def nodes(self) -> list:
        """The nodes in the causal network

        Returns:
            list: The nodes in the causal network

        Example:
            >>> climate_network.nodes
            >>> ['POP', 'EC', 'URB', 'GDP', 'FFEC', 'REC', 'EI', 'CO2', 'CH4', 'N2O']
        """
        return list(self.__bayesian_network.nodes)

    @property
    def edges(self) -> list:
        """The edges in the causal network

        Returns:
            list: The edges in the causal network

        Example:
            >>> print(climate_network.edges)
            >>> [('POP', 'EC'), ('EC', 'FFEC'), ('EC', 'REC'), ('EC', 'EI'), ('URB', 'EC'), ('GDP', 'EC'), ('FFEC', 'CO2'), ('FFEC', 'CH4'), ('FFEC', 'N2O'), ('REC', 'CO2'), ('REC', 'CH4'), ('REC', 'N2O')]
        """
        return list(self.__bayesian_network.edges)

    @property
    def gml_graph(self) -> str:
        """Generates a GML graph that can be fed into a DoWhy CausalModel

        Returns:
            str: the GML graph

        Example:
            >>> from dowhy import CausalModel
            >>> network_1 = CausalNetwork([("X", "Y"), ("W", "X"), ("W", "Y")])
            >>> model_1 = CausalModel(data=data_1, treatment='X', outcome='Y', graph=network_1.gml_graph)
        """
        graph = 'graph [directed 1\n'

        for node in self.nodes:
            graph += f'\tnode [id "{node}" label "{node}"]\n'

        for edge in self.edges:
            graph += f'\tedge [source "{edge[0]}" target "{edge[1]}"]\n'

        graph += ']'

        return graph

    # region old code
    # I used to think you needed a dot_graph for a causal.do but you don't, a gml graph works just fine!!
    # @property
    # def dot_graph(self) -> str:
    #     """Generates a dot graph that can be fed into a DoWhy .do operation in order to feed in a DAG rather than explicitly setting common_causes

    #     Returns:
    #         str: the dot graph

    #     Example:
    #         >>> training_model_edges : list = [("age", "treat"), ("age", "re78"),
    #         >>>                                ("educ", "treat"), ("educ", "re78"),
    #         >>>                                ("black", "treat"), ("black", "re78"),
    #         >>>                                ("hisp", "treat"), ("hisp", "re78"),
    #         >>>                                ("married", "treat"), ("married", "re78"),
    #         >>>                                ("nodegr", "treat"), ("nodegr", "re78"),]

    #         >>> training_model_pos : dict = {"treat": [1,1], "age": [2, 3], "educ": [3, 3], "black": [4, 3], "hisp": [5, 3], "married": [6,3], "nodegr": [7, 3], "re78": [8, 1]}

    #         >>> import dowhy.api

    #         >>> training_model = DirectedAcyclicGraph(edges=training_model_edges)
    #         >>> training_model.display_pgm_model(pos=training_model_pos)

    #         >>> cdf_0 = df.causal.do(x={treatment: 0},
    #         >>>                      variable_types={treatment: 'b', outcome: 'c', common_cause: 'c'},
    #         >>>                      outcome="re78",
    #         >>>                      dot_graph=training_model.dot_graph,
    #         >>>                      proceed_when_unidentifiable=True)

    #     """
    #     dot_graph = Digraph()

    #     for edge in self.edges:
    #         dot_graph.edge(edge[0], edge[1])

    #     return dot_graph.source
    # endregion

    @staticmethod
    def infer_variable_types(data : pd.DataFrame) -> dict:
        """Infers the variable types dictionary that is a required part of the dowhy library do operators

        Args:
            data (pd.DataFrame): The dataframe to be inferred

        Returns:
            dict: A dictionary with the feature names as the keys and the inferred datatypes as the values

        Examples:
            >>> variable_types = infer_variable_types(df_lalonde[features])
            >>> variable_types
                {'treat': 'd',
                'age': 'c',
                'educ': 'c',
                'black': 'd',
                'hisp': 'd',
                'married': 'd',
                'nodegr': 'd',
                're78': 'c'}

            >>> df_do_lalonde_1 = df_lalonde.causal.do(x={"treat": 1},
            >>>                                        outcome="re78",
            >>>                                        dot_graph=training_model.gml_graph,
            >>>                                        variable_types=infer_variable_types,
            >>>                                        proceed_when_unidentifiable=True)

        Notes:
            The dowhy causal.do operator requires an explicit list of variable types for each variable that is used. Those variables
            must include the treatment / x, the outcome and all of the variables / nodes found in the dot_graph or alternatively those
            explictly specified in the common_causes. Features may be present in the dataframe that are not used in the do operation
            and if there are any of those they should be excluded.

            In the dowhy documentation (https://www.pywhy.org/dowhy/v0.8/dowhy.do_samplers.html?highlight=variable_types) it states
            that valid values are "'c' for continuous, 'o' for ordered, 'd' for discrete, and 'u' for unordered".

            Statistically speaking integers are discrete but according to statsexchange (https://stats.stackexchange.com/questions/261396/integer-data-categorical-or-continuous)
            "Integers are discrete, not continuous, but to treat them as nominal categories would throw out most of the information,
            and even treating them as ordinal could lose quite a bit.".

            This makes sense for a causal inference model. In my earlier research with pgmpy I found that integers were represented as
            a massive set of discrete values and this meant that if the test data had gaps between numbers there would be issues with the
            model. Also it makes intuitive sense that the number 25 is not treated any differently to 25.0.

            In the dowhy documentation (https://www.pywhy.org/dowhy/v0.8/example_notebooks/lalonde_pandas_api.html?highlight=do+operation)
            the LaLonde example specifies the variable_types as follows -
            >>> variable_types={'age': 'c', 'educ':'c', 'black': 'd', 'hisp': 'd', 'married': 'd', 'nodegr': 'd','re78': 'c', 'treat': 'b'}
            Note that age and educ are specified as "c" for continuous even though they are both integers with discrete calues.

            This was the final piece of evidence; if age and educ are changed to "d" the do operation produces wildly different results
            which led to me deciding that I needed to follow the stackexchange advice to treat integers as continuous where a linear
            regression calculation was going to be performed.

            With all that in mind the agorithm for infering the variable types became very simple -
            1) IF a variable is a numeric datatype and has more than two unique values it is considered to be continuous
            2) If a variable is a numeric datatype and has less than two unique values e.g. (1, 0; True, False; "yes", "no") it is considered to be discrete.
            3) If a variable is a non-numeric datatype it is considered to be discrete

            This has been tested against the LaLonde data set and it produces the same results as the examples in the dowhy documentation.
            It has also been tested against categorical values ("yes", "no") and that worked as well.

            That testing revealed that dowhy.do does not care whether c variable type is "b" or "d" and in any case "b" is inconsistent
            in that it is not mentioned in the documented types.
        """
        numeric_features : list = list(data.select_dtypes(include=np.number).columns) # Gets a list of the numeric features
        dataframe_features_dict : dict = data.nunique().to_dict() # Gets a dictionary of each feature and the number of unique values found in the data

        return {key: "c" if value > 2 and key in numeric_features else "d" for key, value in dataframe_features_dict.items()}

    @staticmethod
    def discover(data : pd.DataFrame, root_nodes : list, predefined_edges : list, excluded_edges : list, excluded_nodes : list, threshold : float = 0.1, iterations : int = 3) -> list: #pylint: disable="dangerous-default-value"
        """Implements a version of causal discovery based solely on correlation

        Args:
            data (pd.DataFrame): The dataframe to be discovered
            root_nodes (list): Initialise with the "outcome" variable in a list e.g. root_nodes=["is_canceled"]
            predefined_edges (list): Initialise with a list of edges to explictly include in the model or [] if there are no edges to be explictly included
            excluded_edges (list): Initialise with a list of edges to explitly exclude into the model or [] if there are no edges to be explictly excluded
            excluded_nodes (list): Initialise with a list of nodes to explitly exclude into the model or [] if there are no nodes to be explictly excluded
            threshold (float, optional): The correlation value which must be exceeded in order for the edge to be added. Defaults to 0.1.
            iterations (int, optional): The number of times to re-discover the new edges that have been discovered in the first iteration. Defaults to 3.

        Returns:
            list: A list of edges that can be fed back into the constructor of the an object of type DirectedAcyclicGraph

        Examples:
            >>> correlated_edges = DirectedAcyclicGraph.discover(data=df_hotel, root_nodes=["is_canceled"], edges=list(), threshold=0.1, iterations=3)
            >>> dag_discovery = DirectedAcyclicGraph(edges=correlated_edges)
            >>> dag_discovery.display_digraph_model()

            >>> unobserved_confounders = [("U", "required_car_parking_spaces"), ("U", "total_of_special_requests"), ("U", "total_stay"), ("U", "guests")]
            >>> correlated_edges = DirectedAcyclicGraph.discover(data=df_hotel, root_nodes=["is_canceled"], edges=unobserved_confounders, threshold=0.1, iterations=3)
            >>> dag_discovery = DirectedAcyclicGraph(edges=correlated_edges)
            >>> dag_discovery.display_digraph_model()
        """
        if iterations == 0:
            #return predefined_edges
            return [(node, root_node) for (node, root_node) in predefined_edges if node not in excluded_nodes and root_node not in excluded_nodes]

        data_encoded = data.copy(deep=True) # Non-numeric data types must be label encoded so that they can be evaluated using .corr()
        for col in data_encoded.select_dtypes(exclude=[np.number]):
            data_encoded[col] = LabelEncoder().fit_transform(data_encoded[col])

        #for root_node in [root_node for root_node in root_nodes if root_node not in excluded_nodes]:
        for root_node in root_nodes:
            ds_correlated_nodes = data_encoded.corr()[root_node].abs().sort_values(ascending=False)[1:]
            ds_correlated_nodes = ds_correlated_nodes[ds_correlated_nodes >= threshold]

            predefined_edges.extend([(node, root_node) for node in list(ds_correlated_nodes.index) if (root_node, node) not in predefined_edges and (node, root_node) not in predefined_edges and (node, root_node) not in excluded_edges and (root_node, node) not in excluded_edges])

        return DirectedAcyclicGraph.discover(data=data, root_nodes=list(ds_correlated_nodes.index), predefined_edges=predefined_edges, excluded_edges=excluded_edges, excluded_nodes=excluded_nodes, threshold=threshold, iterations=iterations-1)

    def display_daft_model(self, figsize : tuple = (2, 2)):
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
        self.__bayesian_network.to_daft().render()
        set_size(figsize[0], figsize[1])
        plt.show()

    def display_networkx_model(self, pos : dict = None, figsize : tuple = (10, 8), node_size : int = 5000, node_color : str = "", auto_layout_cols : int = 0):
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
            node_pos = [[x, y] for x in range(auto_layout_cols) for y in range(ceil(len(self.__bayesian_network.nodes)/auto_layout_cols))]
            pos = {k: node_pos[i] for i, k in enumerate(list(self.__bayesian_network.nodes))}

        plt.figure(figsize=figsize)
        plt.box(False)
        draw_networkx(DiGraph(self.__bayesian_network.edges), with_labels=True, pos=pos, node_color=[f"C{i}" for i in range(len(self.__bayesian_network.nodes))] if node_color == "" else node_color, node_size=node_size, arrowsize=25)
        plt.show()

    def display_pgm_model(self, pos : dict, node_names : dict =  None, print_node_names : bool = False):
        """Plots a small format diagram that is similar to daft but that allows for (and requires that) the node positions are specified

        Args:
            pos (dict): the positions of the nodes
            node_names (dict, optional): If provided this is a mapping from the node names to what will be displayed on the graph. Defaults to None.
            print_node_names (bool, optional): If True and node_names is provided then the mapping is displayed to aid readability. Defaults to False.

        Example:
            >>> edges : list = [("title_length", "click_through_rate"), ("author", "title_length"), ("author", "click_through_rate")]
            >>> pos : dict = {"title_length": [1, 1], "click_through_rate": [3, 1], "author": [2, 2]}
            >>> node_names : dict = {"title_length" : "T", "click_through_rate" : "C", "author" : "A"}

            >>> dag = DirectedAcyclicGraph(edges=edges)
            >>> dag.display_pgm_model(pos=pos, node_names=node_names, print_node_names=True)
        """
        max_x = max([x[0] for x in list(pos.values())])
        max_y = max([y[1] for y in list(pos.values())])

        pgm = PGM(shape=[max_x, max_y])

        node_names = {node_name: node_name for node_name in self.nodes} if node_names is None else node_names

        if print_node_names and node_names is not None:
            print_node_names : str = ""
            for node_name, node_id in node_names.items():
                print_node_names += f"{node_id} = {node_name}, "
            print(print_node_names[:-2])

        for node in pos.keys():
            pgm.add_node(daft.Node(node, node_names[node], pos[node][0], pos[node][1]))

        for edge in self.edges:
            pgm.add_edge(edge[0], edge[1])

        pgm.render()
        plt.show()

    #def display_digraph_model(self, figsize : tuple = (500, 500), notebook : bool = True, filename : str = "digraph.svg") -> IFrame:
    #        figsize (tuple, optional): _description_. Defaults to (500, 500).
    def display_digraph_model(self, notebook : bool = True, filename : str = "digraph.svg") -> IFrame:
        """Plots a cartoon-like directed acyclic graph with eliptical bubbles for nodes and an intuitive and readable layout

        Args:
            model (BayesianNetwork): The model to be drawn which must implement an edges collection
            notebook (bool, optional): If True the graph is displayed inside a Jupyter Notebook cell, if false a PDF document is launched in a separate browser window. Defaults to True
            filename (str, optional): The temporary filename used to save and store the HTML output. Defaults to "digraph.svg". This parameter also controls the format with "svg" being the best in terms of resolution.

        Returns:
            IFrame: An IFrame is returned if notebook = True so that it can be embedded in the Jupyter Notebook cell. If notebook = False there is no return value as the graph is rendered in a separate browser window

        Examples
            >>> from pgmpy.models import BayesianNetwork
            >>> domain_model = BayesianNetwork([('Vaccination?', 'Reaction?'), ('Vaccination?', 'Smallpox?'), ('Reaction?', 'Death?'), ('Smallpox?', 'Death?')])

            >>> # Display within the Jupyter Notebook cell
            >>> display_digraph_model(model=model, figsize=(400,400))
        """
        digraph = Digraph()
        digraph.edges(list(self.__bayesian_network.edges))

        if notebook:
            digraph.render(filename.split('.')[0], format=filename.split('.')[1])
            #return IFrame(src=filename, width=figsize[0], height=figsize[1])
            #return IFrame(src=Image(filename, width=figsize[0], height=figsize[1]), width=figsize[0], height=figsize[1])
            display(SVG(filename))
        else:
            digraph.view()

    def display_pyvis_model(self, figsize : tuple = (500, 500), notebook : bool = True, enable_physics : bool = False, hierarchical : bool = False, filename : str = "pyvis.html") -> IFrame:
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

        References:
            # https://towardsdatascience.com/making-network-graphs-interactive-with-python-and-pyvis-b754c22c270
            # https://github.com/WestHealth/pyvis/issues/48
        """
        net = Network(height=f"{figsize[0]}px", width=f"{figsize[1]}px", notebook=notebook, directed=True, heading="") # Set up the Network object

        net.from_nx(DiGraph(self.__bayesian_network.edges)) # Build the nodes and edges directly from the BayesianNetwork

        if notebook:
            options = 'var options = {"edges": {"color": {"inherit": true},"smooth": false}'
            if not enable_physics:
                options = options + ',"physics": {"enabled": false,"minVelocity": 0.75}'
            if hierarchical:
                options = options + ',"layout": {"hierarchical": {"enabled": true}}'
            options = options + "}"

            net.set_options(options)
        else:
            net.show_buttons()

        net.show(filename) # Create the temporary file that contains the HTML and CSS

        if notebook: # If displaying and embedding in a Jupyter Notebook cell an IFrame must be returned or nothing will be rendered
            return IFrame(src=filename, width=figsize[0]*1.1, height=figsize[1]*1.1)




# """This module contains helpder functions to visualise directed acyclic graphs
# """
# from math import ceil
# import matplotlib.pyplot as plt
# from daft import PGM
# from IPython.display import IFrame

# from networkx import draw_networkx, DiGraph
# from pgmpy.models import BayesianNetwork
# from pyvis.network import Network
# from graphviz import Digraph

# def __set_size(width : int ,height : int):
#     """Explicitly sets the size of a matplotlib plot

#     Args:
#         width (int): Width in inches
#         height (int): Height in inches

#     References:
#         https://stackoverflow.com/questions/44970010/axes-class-set-explicitly-size-width-height-of-axes-in-given-units
#     """
#     axis=plt.gca()

#     figw = float(width)/(axis.figure.subplotpars.right - axis.figure.subplotpars.left)
#     figh = float(height)/(axis.figure.subplotpars.top-axis.figure.subplotpars.bottom)

#     axis.figure.set_size_inches(figw, figh)

# def display_daft_model(model : BayesianNetwork, figsize : tuple = (2, 2)) -> PGM:
#     """Plots a small-format directed acyclic graph with black-and-white nodes that can be positioned

#     Args:
#         model (BayesianNetwork): The model to be drawn which must implement an edges collection
#         pos (dict, optional): A dictionary with nodes as keys and positions as values. Each value is a list containing an X and Y co-ordinate. If not specified a spring layout positioning will be computed. Defaults to None (automatic layout).
#         figsize (tuple, optional): Sets the figure size. If None default (small) sizing is used. Defaults to (2, 2).

#     Example:
#         >>> from pgmpy.models import BayesianNetwork

#         >>> domain_model = BayesianNetwork([('Vaccination?', 'Reaction?'), ('Vaccination?', 'Smallpox?'), ('Reaction?', 'Death?'), ('Smallpox?', 'Death?')])
#         >>> plot_daft_model(domain_model, pos=POS)
#     """
#     model.to_daft().render()
#     __set_size(figsize[0], figsize[1])
#     plt.show()

# def display_networkx_model(model : BayesianNetwork, pos : dict = None, figsize : tuple = (10, 8), node_size : int = 5000, auto_layout_cols : int = 0):
#     """Plots a large-format directed acyclic graph with coloured nodes that can be sized and positioned

#     Args:
#         model (BayesianNetwork): The model to be drawn which must implement an edges collection
#         pos (dict, optional): A dictionary with nodes as keys and positions as values. Each value is a list containing an X and Y co-ordinate. If not specified a spring layout positioning will be computed. Defaults to None (automatic layout).
#         figsize (tuple, optional): The size of the displayed plot. Defaults to (10, 8).
#         node_size (int, optional): The size of the nodes in the plot. Defaults to 5000.
#         auto_layout_cols (int, optional): If set > 0 an auto-layout is generated with the specified number of columns. Defaults to 0.

#     Example:
#         >>> from pgmpy.models import BayesianNetwork

#         >>> POS : dict = {'Vaccination?': [0, 1], 'Reaction?': [-1, 0], 'Smallpox?': [1, 0], 'Death?': [0, -1]}
#         >>> domain_model = BayesianNetwork([('Vaccination?', 'Reaction?'), ('Vaccination?', 'Smallpox?'), ('Reaction?', 'Death?'), ('Smallpox?', 'Death?')])
#         >>> plot_model(model=domain_model, pos=POS)
#     """
#     if auto_layout_cols > 0:
#         node_pos = [[x, y] for x in range(auto_layout_cols) for y in range(ceil(len(model.nodes)/auto_layout_cols))]
#         pos = {k: node_pos[i] for i, k in enumerate(list(model.nodes))}

#     plt.figure(figsize=figsize)
#     plt.box(False)
#     draw_networkx(DiGraph(model.edges), with_labels=True, pos=pos, node_color=[f"C{i}" for i in range(len(model.nodes))], node_size=node_size, arrowsize=25)
#     plt.show()

# def display_digraph_model(model : BayesianNetwork, figsize : tuple = (500, 500), notebook : bool = True, filename : str = "digraph.svg") -> IFrame:
#     """Plots a cartoon-like directed acyclic graph with eliptical bubbles for nodes and an intuitive and readable layout

#     Args:
#         model (BayesianNetwork): The model to be drawn which must implement an edges collection
#         figsize (tuple, optional): _description_. Defaults to (500, 500).
#         notebook (bool, optional): If True the graph is displayed inside a Jupyter Notebook cell, if false a PDF document is launched in a separate browser window. Defaults to True
#         filename (str, optional): The temporary filename used to save and store the HTML output. Defaults to "digraph.svg". This parameter also controls the format with "svg" being the best in terms of resolution.

#     Returns:
#         IFrame: An IFrame is returned if notebook = True so that it can be embedded in the Jupyter Notebook cell. If notebook = False there is no return value as the graph is rendered in a separate browser window

#     Examples
#         >>> from pgmpy.models import BayesianNetwork
#         >>> domain_model = BayesianNetwork([('Vaccination?', 'Reaction?'), ('Vaccination?', 'Smallpox?'), ('Reaction?', 'Death?'), ('Smallpox?', 'Death?')])

#         >>> # Display within the Jupyter Notebook cell
#         >>> display_digraph_model(model=model, figsize=(400,400))
#     """
#     digraph = Digraph()
#     digraph.edges(list(model.edges))

#     if notebook:
#         digraph.render(filename.split('.')[0], format=filename.split('.')[1])
#         return IFrame(src=filename, width=figsize[0], height=figsize[1])
#     else:
#         digraph.view()

# def display_pyvis_model(model : BayesianNetwork, figsize : tuple = (500, 500), notebook : bool = True, enable_physics : bool = False, hierarchical : bool = False, filename : str = "pyvis.html") -> IFrame:
#     """Displays a fully interactive directed acyclic graph that can either be embedded in a Jupyter Notebook cell or displayed in a new browser window

#     Args:
#         model (BayesianNetwork): The model to be drawn which must implement an edges collection
#         figsize (tuple, optional): The size of the displayed plot. Defaults to (10, 8)
#         notebook (bool, optional): If True the graph is displayed inside a Jupyter Notebook cell, if false it is launched in a separate browser window. Defaults to True
#         enable_physics (bool, optional): If True the nodes "jiggle" when they are dragged which looks really neat but it can mean that it is difficult to get a good lauout. False turns the "jiggle" off. Defaults to False.
#         filename (str, optional): The temporary filename used to save and store the HTML output. Defaults to "pyvis.html"

#     Returns:
#         IFrame: An IFrame is returned if notebook = True so that it can be embedded in the Jupyter Notebook cell. If notebook = False there is no return value as the graph is rendered in a separate browser window

#     Examples:
#         >>> from pgmpy.models import BayesianNetwork
#         >>> domain_model = BayesianNetwork([('Vaccination?', 'Reaction?'), ('Vaccination?', 'Smallpox?'), ('Reaction?', 'Death?'), ('Smallpox?', 'Death?')])

#         >>> # Display within the Jupyter Notebook cell
#         >>> display_pyvis_model(model=model)

#         >>> # Display in a new browser window (including all menus)
#         >>> display_pyvis_model(model=model, notebook=False)

#         >>> # Display within the Jupyter Notebook cell and make the default size larger by 50%
#         >>> display_pyvis_model(model=model, figsize=(750, 750))
#     """
#     net = Network(height=f"{figsize[0]}px", width=f"{figsize[1]}px", notebook=notebook, directed=True, heading="") # Set up the Network object
#     #if not notebook: # If displaying in a separate browser window ensure the buttons window is displayed
#     #    net.show_buttons()

#     # https://towardsdatascience.com/making-network-graphs-interactive-with-python-and-pyvis-b754c22c270
#     # https://github.com/WestHealth/pyvis/issues/48

#     net.from_nx(DiGraph(model.edges)) # Build the nodes and edges directly from the BayesianNetwork

#     if notebook:
#         options = 'var options = {"edges": {"color": {"inherit": true},"smooth": false}'
#         if not enable_physics:
#             options = options + ',"physics": {"enabled": false,"minVelocity": 0.75}'
#         if hierarchical:
#             options = options + ',"layout": {"hierarchical": {"enabled": true}}'
#         options = options + "}"

#         # if not enable_physics:
#         #     options = r'var options = {"edges": {"color": {"inherit": true},"smooth": false},"physics": {"enabled": false,"minVelocity": 0.75}}'

#         #     net.set_options(options)

#         net.set_options(options)
#     else:
#         net.show_buttons()

#     net.show(filename) # Create the temporary file that contains the HTML and CSS

#     if notebook: # If displaying and embedding in a Jupyter Notebook cell an IFrame must be returned or nothing will be rendered
#         return IFrame(src=filename, width=figsize[0]*1.1, height=figsize[1]*1.1)

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
