import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def conditional_statistical_parity_test(df, outcome_col, protected_attr, conditioned_attr, positive_outcome=1):
    """
    Performs Conditional Statistical Parity Test.

    Parameters:
    - df: DataFrame containing the data.
    - outcome_col: Column name of model predictions or actual outcomes (e.g., 'y_pred').
    - protected_attr: Column name of the protected attribute (e.g., 'gender' or 'race').
    - conditioned_attr: Column name of the legitimate conditioning attribute (e.g., 'education_level').
    - positive_outcome: The value considered as a positive prediction (default: 1).

    Returns:
    - A DataFrame showing the positive outcome rate for each group (protected vs non-protected) after conditioning.
    """
    # Group by both the protected attribute and the legitimate conditioned attribute
    grouped = df.groupby([conditioned_attr, protected_attr])
    
    # Initialize result dictionary
    result = {
        'Condition': [],
        'Protected_Group': [],
        'Positive_Outcome_Rate': [],
        'Total_Samples': []
    }

    for group_vals, group_data in grouped:
        conditioned_value, protected_value = group_vals
        total_samples = len(group_data)
        
        # Calculate the positive outcome rate in each group
        positive_outcomes = np.sum(group_data[outcome_col] == positive_outcome)
        positive_rate = positive_outcomes / total_samples if total_samples > 0 else 0
        
        # Append results for this group
        result['Condition'].append(conditioned_value)
        result['Protected_Group'].append(protected_value)
        result['Positive_Outcome_Rate'].append(positive_rate)
        result['Total_Samples'].append(total_samples)
    
    # Convert result to DataFrame
    result_df = pd.DataFrame(result)
    
    return result_df