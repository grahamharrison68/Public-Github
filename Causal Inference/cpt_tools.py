"""This module contains helpder functions to visualise conditional probability tables
"""
from itertools import product
import pandas as pd
import numpy as np
from IPython.display import display

from pgmpy.factors.discrete import TabularCPD, DiscreteFactor

def discrete_factor_to_dataframe(discrete_factor: DiscreteFactor) -> pd.DataFrame:
    """A modification of the code from DiscreteFactor._str which takes the printout from a DiscreteFactor and converts it into a DataFrame

    Args:
        discrete_factor (DiscreteFactor): The DiscreteFactor to be converted

    Returns:
        pd.DataFrame: The discrete factor table formatted as a DataFrame
    """

    string_header = list(map(str, discrete_factor.scope()))
    string_header.append(f"{'p'}({','.join(string_header)})")

    value_index = 0
    factor_table = []
    for prob in product(*[range(card) for card in discrete_factor.cardinality]):
        if discrete_factor.state_names:
            prob_list = [
                "{var}({state})".format(
                    var=list(discrete_factor.variables)[i],
                    state=discrete_factor.state_names[list(discrete_factor.variables)[i]][prob[i]],
                )
                for i in range(len(discrete_factor.variables))
            ]
            prob_list = [discrete_factor.state_names[list(discrete_factor.variables)[i]][prob[i]] for i in range(len(discrete_factor.variables))]
        else:
            prob_list = [
                f"{list(discrete_factor.variables)[i]}_{prob[i]}"
                for i in range(len(discrete_factor.variables))
            ]

        prob_list.append(discrete_factor.values.ravel()[value_index])
        factor_table.append(prob_list)
        value_index += 1

    return pd.DataFrame(data=factor_table, columns=string_header)


# def display_discrete_factor(discrete_factor : DiscreteFactor) -> pd.DataFrame:
#     """Displays a pgmpy DiscreteFactor to a Jupyter cell output 

#     Args:
#         discrete_factor (DiscreteFactor): The DiscreteFactor to be displayed.

#     Returns:
#         pd.DataFrame: The discrete factor table formatted as a DataFrame
#     """
#     df_discrete_factor = __discrete_factor_to_dataframe(discrete_factor)

#     print("hello")
#     display(df_discrete_factor)

#     return df_discrete_factor


    #return tabulate(
    #    factor_table, headers=string_header, tablefmt=tablefmt, floatfmt=".4f"
    #)




    # def _str(self, phi_or_p="phi", tablefmt="grid", print_state_names=True):
    #     """
    #     Generate the string from `__str__` method.

    #     Parameters
    #     ----------
    #     phi_or_p: 'phi' | 'p'
    #             'phi': When used for Factors.
    #               'p': When used for CPDs.
    #     print_state_names: boolean
    #             If True, the user defined state names are displayed.
    #     """
    #     string_header = list(map(str, self.scope()))
    #     string_header.append(f"{phi_or_p}({','.join(string_header)})")

    #     value_index = 0
    #     factor_table = []
    #     for prob in product(*[range(card) for card in self.cardinality]):
    #         if self.state_names and print_state_names:
    #             prob_list = [
    #                 "{var}({state})".format(
    #                     var=list(self.variables)[i],
    #                     state=self.state_names[list(self.variables)[i]][prob[i]],
    #                 )
    #                 for i in range(len(self.variables))
    #             ]
    #         else:
    #             prob_list = [
    #                 f"{list(self.variables)[i]}_{prob[i]}"
    #                 for i in range(len(self.variables))
    #             ]

    #         prob_list.append(self.values.ravel()[value_index])
    #         factor_table.append(prob_list)
    #         value_index += 1

    #     return tabulate(
    #         factor_table, headers=string_header, tablefmt=tablefmt, floatfmt=".4f"
    #     )




def __cpt_to_dataframe(tabular_cpd : TabularCPD) -> pd.DataFrame:
    """Converts a TabularCPD (pgmpy Conditional Probability Table object) output into a DataFrame

    Args:
        tabularcpd (tabularcpd): The TabularCPD to be converted.

    Returns:
        pd.DataFrame: The output DataFrame

    Notes:
        Modified from _make_table_str in CPD.py (right click on TabularCPD and choose Goto Definition)
    """
    headers_list = []

    # Build column headers
    evidence = tabular_cpd.variables[1:]
    evidence_card = tabular_cpd.cardinality[1:]
    if evidence:
        col_indexes = np.array(list(product(*[range(i) for i in evidence_card])))
        if tabular_cpd.state_names:
            for i in range(len(evidence_card)):
                column_header = [str(evidence[i])] + [
                    "{var}({state})".format(
                        var=evidence[i], state=tabular_cpd.state_names[evidence[i]][d]
                    )
                    for d in col_indexes.T[i]
                ]
                headers_list.append(column_header)
        else:
            for i in range(len(evidence_card)):
                column_header = [str(evidence[i])] + [
                    f"{evidence[i]}_{d}" for d in col_indexes.T[i]
                ]
                headers_list.append(column_header)

    # Build row headers
    if tabular_cpd.state_names:
        variable_array = [
            [
                "{var}({state})".format(
                    var=tabular_cpd.variable, state=tabular_cpd.state_names[tabular_cpd.variable][i]
                )
                for i in range(tabular_cpd.variable_card)
            ]
        ]
    else:
        variable_array = [
            [f"{tabular_cpd.variable}_{i}" for i in range(tabular_cpd.variable_card)]
        ]
    # Stack with data
    labeled_rows = np.hstack(
        (np.array(variable_array).T, tabular_cpd.get_values())
    ).tolist()

    # If there is no headers list (because there is only one variable) then create one ready for the DataFrame conversion
    if headers_list == []:
        headers_list = [[tabular_cpd.variables[0], "P"]]

    # Convert the data and headers arrays into a DataFrame
    df_cpd = pd.DataFrame(data=labeled_rows, columns=headers_list)

    # If there is only one variable this is a "header node" i.e. one with no inputs / parents so the default format provided looks best
    # If there are multiple variables i.e. a node with one or more inputs / parents then transposing the DataFrame give an output format
    # that closely resembles that proposed by Pearl in "The Book of Why", Chapter 3, p117
    if len(tabular_cpd.variables) > 1:
        df_cpd = df_cpd.T
        df_cpd.columns = df_cpd.iloc[0]
        df_cpd = df_cpd.iloc[1: , :]

    df_cpd = df_cpd.apply(pd.to_numeric, errors="ignore")
    return df_cpd

def display_cpt(tabularcpd : TabularCPD) -> pd.DataFrame:
    """Displays a CPD to a Jupyter cell output including a formatted explanation and a DataFram of the probabilities

    Args:
        tabularcpd (tabularcpd): The TabularCPD to be displayed.

    Returns:
        pd.DataFrame: The conditional probability table formatted as a DataFrame
    """
    description = f"Probability of {tabularcpd.variables[0]}"
    if len(tabularcpd.variables) > 1:
        description = description + f" given {', '.join(tabularcpd.variables[1:])}"

    df_cpd = __cpt_to_dataframe(tabularcpd)

    print(description)
    display(df_cpd)

    return df_cpd

def display_query(query : DiscreteFactor, phi_or_p : str = "phi", print_state_names : bool = True) -> pd.DataFrame:
    """Converts a DiscreteFactor (a pgmpy query object) output into a DataFrame

    Args:
        phi_or_p (str, optional): 'phi': When used for Factors. 'p': When used for CPDs. Defaults to "phi".
        print_state_names (bool, optional): If True, the user defined state names are displayed.. Defaults to True.

    Returns:
        pd.DataFrame: The output DataFrame

    Notes:
        Modified from _str in DiscreteFactor.py (right click on TabularCPD and choose Goto Definition)
    """
    string_header = list(map(str, query.scope()))
    string_header.append(f"{phi_or_p}({','.join(string_header)})")

    value_index = 0
    factor_table = []
    for prob in product(*[range(card) for card in query.cardinality]):
        if query.state_names and print_state_names:
            prob_list = [
                "{var}({state})".format(
                    var=list(query.variables)[i],
                    state=query.state_names[list(query.variables)[i]][prob[i]],
                )
                for i in range(len(query.variables))
            ]
        else:
            prob_list = [
                f"{list(query.variables)[i]}_{prob[i]}"
                for i in range(len(query.variables))
            ]

        prob_list.append(query.values.ravel()[value_index])
        factor_table.append(prob_list)
        value_index += 1

    return pd.DataFrame(factor_table, columns=string_header)
