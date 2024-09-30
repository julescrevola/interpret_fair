from scipy.stats import chi2_contingency
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def test_statistical_parity(variable_to_test: pd.Series, outcome: pd.Series) -> float:
    """Run statistical parity test, ensuring the contingency table is valid."""
    contingency_table = pd.crosstab(variable_to_test, outcome)

    # Ensure we have enough data in the contingency table
    if contingency_table.min().min() < 5:
        # If any cell has fewer than 5 observations, return NaN as the result is unreliable
        return np.nan
    
    # Calculate the chi-squared statistic and p-value
    chi2, p, _, _ = chi2_contingency(contingency_table)

    return p

def calculate_p_values(variable_to_test: pd.Series, outcome: pd.Series, bin_edges: np.ndarray) -> np.ndarray:
    """Calculate p-values for different variable thresholds based on bin ranges."""
    p_values = []
    for threshold in bin_edges:
        # Bin the variable_to_test based on the threshold
        binned_variable = (variable_to_test >= threshold).astype(int)

        # Calculate the p-value for the statistical parity test
        p = test_statistical_parity(binned_variable, outcome)
        p_values.append(p)
    return np.array(p_values)

def draw_fpdp(variable_to_test: pd.Series, outcome: pd.Series):
    """Draw FPDP graphs with the p-values of the statistical parity test."""
    # Bin the variable_to_test into 10 ranges (for continuous variables)
    _, bin_edges = pd.qcut(variable_to_test, 10, retbins=True, duplicates='drop')

    # Calculate p-values for each threshold (bin edge) on the variable_to_test
    p_values = calculate_p_values(variable_to_test, outcome, bin_edges)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(bin_edges, p_values, marker='o', label='P-value')
    plt.axhline(y=0.05, color='r', linestyle='--', label='Significance Level (α = 0.05)')
    plt.title('P-value vs. Variable Thresholds (Binned into 10 ranges)')
    plt.xlabel('Variable Value Threshold')
    plt.ylabel('P-value')
    plt.xticks(bin_edges, rotation=45)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.show()
