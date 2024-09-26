"""Module for testing statistical_parity"""

from scipy.stats import chi2_contingency
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def test_statistical_parity(variable_to_test:pd.Series, outcome:pd.Series) -> float:
    """Run statistical parity test"""
    contingency_table = pd.crosstab(variable_to_test, outcome)

    # Calculate the chi-squared statistic and p-value
    chi2, p, _, _ = chi2_contingency(contingency_table)

    return p

def calculate_p_values(variable_to_test: pd.Series, outcome: pd.Series, thresholds: np.ndarray) -> np.ndarray:
    "Append p values based on threshold"
    p_values = []
    for threshold in thresholds:
        # Bin the outcomes based on the threshold
        binned_outcome = (outcome >= threshold).astype(int)
        p = test_statistical_parity(variable_to_test, binned_outcome)
        p_values.append(p)
    return np.array(p_values)

def draw_fpdp(variable_to_test: pd.Series, outcome: pd.Series):
    """Draw fpdp graphs"""
    thresholds = np.unique(outcome)

    # Calculate p-values for different thresholds
    p_values = calculate_p_values(variable_to_test, outcome, thresholds)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, p_values, marker='o')
    plt.axhline(y=0.05, color='r', linestyle='--', label='Significance Level (α = 0.05)')
    plt.title('P-value vs. Outcome Thresholds')
    plt.xlabel('Outcome Value Threshold')
    plt.ylabel('P-value')
    plt.xticks(thresholds)
    plt.ylim(0, 1)
    plt.grid()
    plt.legend()
    plt.show()



