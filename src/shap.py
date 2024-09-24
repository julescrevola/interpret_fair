"""Function to evaluate SHAP for a specific model"""

import shap
import matplotlib.pyplot as plt
import pandas as pd
from typing import Literal

from sklearn.base import RegressorMixin, ClassifierMixin, BaseEstimator


def draw_shap(model: RegressorMixin | ClassifierMixin | BaseEstimator,
              data: pd.DataFrame, type: Literal["bar"],
              model_name: str | None = None):
    """Plot shap values"""

    if model_name is None:
        model_name = type(model).__name__

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)

    shap.summary_plot(shap_values, data, show=False, plot_type=type)

    plt.savefig(f"variables_impact for {model_name}", dpi=200)

    plt.show()


def draw_shap_tree(model: RegressorMixin | ClassifierMixin | BaseEstimator,
                   data: pd.DataFrame, model_name: str | None = None):
    """Plot shap tree for variables"""
    draw_shap(model=model,
              data=data,
              model_name=model_name,
              type="bar")


def draw_shap_summary(model: RegressorMixin | ClassifierMixin | BaseEstimator,
                      data: pd.DataFrame, model_name: str | None = None):
    """Plot shap summary for variables"""
    draw_shap(model=model,
              data=data,
              model_name=model_name,
              type="bar")
