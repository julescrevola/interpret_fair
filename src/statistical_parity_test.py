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

def calculate_p_values(variable_to_test: pd.Series, outcome: pd.Series) -> np.ndarray:
    """Calculate p-values for different variable thresholds based on unique values or binned ranges."""
    unique_values = variable_to_test.nunique()
    
    if unique_values > 10:
        # Bin the variable into 10 ranges (percentiles) if there are more than 10 unique values
        # This creates 10 distinct groups rather than a binary split
        binned_variable, bin_edges = pd.qcut(variable_to_test, 10, retbins=True, labels=False, duplicates='drop')
        thresholds = bin_edges
    else:
        # If 10 or fewer unique values, use the unique values directly as groups
        binned_variable = variable_to_test
        thresholds = sorted(variable_to_test.unique())

    # Calculate p-value for the statistical parity test
    p = test_statistical_parity(binned_variable, outcome)
    
    return p, thresholds

def draw_fpdp(variable_to_test: pd.Series, outcome: pd.Series):
    """Draw FPDP graphs with the p-values of the statistical parity test."""
    
    # Calculate p-values for the grouped variable_to_test
    p_value, thresholds = calculate_p_values(variable_to_test, outcome)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, [p_value] * len(thresholds), marker='o', label='P-value')
    plt.axhline(y=0.05, color='r', linestyle='--', label='Significance Level (α = 0.05)')
    plt.title('P-value vs. Variable Groups')
    plt.xlabel('Variable Value Groups')
    plt.ylabel('P-value')
    plt.xticks(thresholds, rotation=45)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.show()
