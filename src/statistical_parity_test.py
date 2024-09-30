import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

from typing import Tuple


def calculate_statistical_parity_p_value(group1_success, group1_total,
                                         group2_success, group2_total):
    """Calculate the p-value for statistical parity between group1 and group2."""
    # Create a contingency table
    contingency_table = np.array([[group1_success,
                                   group1_total - group1_success],
                                  [group2_success,
                                   group2_total - group2_success]])

    # Perform the chi-squared test
    chi2, p_value, _, _ = chi2_contingency(contingency_table, correction=False)

    return p_value


def test_statistical_parity(df: pd.DataFrame,
        binned_variable_name: str,
        binned_edges: np.array,
        group_variable: str, outcome_variable: str) -> float:
    """Test statistical parity by comparing each group to the rest."""
    p_values = []
    df[binned_variable_name] = df[binned_variable_name].astype(int)

    candidate_feature_values = binned_edges
    for i in range(len(candidate_feature_values) - 1):
        bin_lower_bound = int(candidate_feature_values[i])
        bin_upper_bound = int(candidate_feature_values[i + 1])


        data_in_bin = df[(df[f'{binned_variable_name}'] >= bin_lower_bound) & (df[f'{binned_variable_name}'] < bin_upper_bound)]


        outcome = df.loc[data_in_bin.index, outcome_variable]

        group1_mask = (data_in_bin[group_variable] == 0)  # no default
        group2_mask = ~group1_mask

        # Group 1: The group being tested
        group1_total = group1_mask.sum()
        group1_success = outcome[group1_mask].sum()

        # Group 2: All other groups combined
        group2_total = group2_mask.sum()
        group2_success = outcome[group2_mask].sum()

        if group1_total < 5 or group2_total < 5:
            # If there are not enough observations, the result is unreliable
            p_values.append(np.nan)
        else:
            # Calculate the p-value for statistical parity
            p_value = calculate_statistical_parity_p_value(group1_success,
                                                           group1_total,
                                                           group2_success,
                                                           group2_total)
            p_values.append(p_value)

    return np.array(p_values)


def calculate_p_values(variable_to_test: pd.Series,
                       df: pd.DataFrame,
                       group_variable: str,
                       outcome_variable: str
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate p-values for the binned or discrete values of the variable_to_test."""
    unique_values = variable_to_test.nunique()
    binned_variable_name = variable_to_test.name

    if True:
        # Bin the variable into 10 groups if there are more than
        # 10 unique values
        binned_variable, bin_edges = pd.qcut(
            variable_to_test, 10, retbins=True, labels=False, duplicates='drop')
        thresholds = bin_edges
       #print(bin_edges)
    else:
        # If 10 or fewer unique values, use the unique values directly
        binned_variable = variable_to_test
    
        #print(binned_variable)
        thresholds = sorted(variable_to_test.unique())

    # Calculate p-values for each group
    p_values = test_statistical_parity(df=df, 
                                       binned_variable_name=binned_variable_name,
                                       binned_edges = bin_edges, 
                                       group_variable= group_variable, 
                                       outcome_variable = outcome_variable)

    return p_values, thresholds


def draw_fpdp(variable_to_test: pd.Series, group_variable: str, outcome_variable: str, df: pd.DataFrame):
    """Draw FPDP graphs with the p-values of the statistical parity test."""

    # Calculate p-values for the grouped variable_to_test
    p_values, thresholds = calculate_p_values(df=df,
                                              variable_to_test=variable_to_test, 
                                              group_variable=group_variable, 
                                              outcome_variable=outcome_variable)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds[:-1], p_values, marker='o', label='P-value')
    plt.axhline(y=0.05, color='r', linestyle='--',
                label='Significance Level (α = 0.05)')
    plt.title('P-value vs. Variable Groups')
    plt.xlabel('Variable Value Groups')
    plt.ylabel('P-value')
    plt.xticks(thresholds, rotation=45)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.show()