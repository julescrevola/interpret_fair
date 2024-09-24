"""Module that contains PDP and ICE Method"""

from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import pandas as pd
from typing import List

from sklearn.base import RegressorMixin, ClassifierMixin, BaseEstimator


def draw_pdp_ice_graphs(model:RegressorMixin | ClassifierMixin | BaseEstimator, data:pd.DataFrame, variables:List[str], model_name: str | None = None):
    """Plot pdp grpahs and ice for variables"""

    if model_name is None:
        model_name = type(model).__name__
    
    fig, ax = plt.subplots(figsize=(12, 6))

    CB_disp = PartialDependenceDisplay.from_estimator(model, data, variables,
                                                    pd_line_kw = {"color": "red"},
                                                    ice_lines_kw = {"color": "steelblue"},
                                                    kind = 'both', 
                                                    target = 0,
                                                    response_method = 'auto',
                                                    ax=ax)

    plt.title(f'{model_name} Dependency Plots', fontsize=16)

    #plt.savefig("explainability", dpi=200)

    plt.show()

def draw_pdp_graphs(model:RegressorMixin | ClassifierMixin | BaseEstimator, data:pd.DataFrame, variables:List[str], model_name: str | None = None):
    """Plot pdp grpahs and ice for variables"""

    if model_name is None:
        model_name = type(model).__name__
    
    fig, ax = plt.subplots(figsize=(12, 6))

    CB_disp = PartialDependenceDisplay.from_estimator(model, data, variables,
                                                    pd_line_kw = {"color": "red"},
                                                    kind = 'both', 
                                                    target = 0,
                                                    response_method = 'auto',
                                                    ax=ax)

    plt.title(f'{model_name} Dependency Plots', fontsize=16)

    #plt.savefig("explainability", dpi=200)

    plt.show()

def draw_ice_graphs(model:RegressorMixin | ClassifierMixin | BaseEstimator, data:pd.DataFrame, variables:List[str], model_name: str | None = None):
    """Plot pdp grpahs and ice for variables"""

    if model_name is None:
        model_name = type(model).__name__
    
    fig, ax = plt.subplots(figsize=(12, 6))

    CB_disp = PartialDependenceDisplay.from_estimator(model, data, variables,
                                                    ice_lines_kw = {"color": "steelblue"},
                                                    kind = 'both', 
                                                    target = 0,
                                                    response_method = 'auto',
                                                    ax=ax)

    plt.title(f'{model_name} Dependency Plots', fontsize=16)

    #plt.savefig("explainability", dpi=200)

    plt.show()