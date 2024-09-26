import XPER
from XPER.datasets.load_data import loan_status
import pandas as pd
from sklearn.model_selection import train_test_split
from XPER.compute.Performance import ModelPerformance
from XPER.viz.Visualisation import visualizationClass as viz 
from sklearn.inspection import permutation_importance

def xper_method(model, eval_metric, dataset: pd.DataFrame, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
    
    """
    use XPER method to compute model performance and visualize outcome
    """
    
    # Evaluate the model performance using the specified metric(s)
    XPER_ = ModelPerformance(X_train.values, y_train.values, X_test.values, y_test.values, model)
    PM = XPER_.evaluate([eval_metric])

    # Print the performance metrics
    print("Performance Metrics: ", round(PM, 3))

    # Option 1 - Kernel True (more than 10 variables)
    # Calculate XPER values for the model's performance
    XPER_values = XPER_.calculate_XPER_values([eval_metric])

    labels = list(X_train.columns)

    #bar plot
    viz.bar_plot(XPER_values=XPER_values, X_test=X_test, labels=labels, p=6,percentage=True)

    #beeswarn plot
    viz.beeswarn_plot(XPER_values=XPER_values,X_test=X_test, labels=labels)

    #force plot
    viz.force_plot(XPER_values=XPER_values, instance=1, X_test=X_test, variable_name=labels, figsize=(16,4))



#Permutation importance method
from sklearn.inspection import permutation_importance
scoring = ['r2'] #more scoring metrics can be added
r_multi = permutation_importance(
    model, X_train, y_train, n_repeats=30, random_state=0, scoring=scoring)

for metric in r_multi:
    print(f"{metric}")
    r = r_multi[metric]
    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(f"    {X_train.columns[i]:<8}"
                  f"{r.importances_mean[i]:.3f}"
                  f" +/- {r.importances_std[i]:.3f}")