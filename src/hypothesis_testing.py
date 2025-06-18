# src/hypothesis_testing.py
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

def prepare_data(df):
    df['Gender'] = df['Gender'].str.upper().replace({'MALE': 'M', 'FEMALE': 'F'}).fillna('U')
    df['ClaimFrequency'] = df.groupby('PolicyID')['TotalClaims'].transform('sum') / df.groupby('PolicyID')['TransactionMonth'].nunique()
    df['ClaimFrequency'] = df['ClaimFrequency'].fillna(0)
    df['HasClaim'] = (df['TotalClaims'] > 0).astype(int)
    df['ClaimSeverity'] = df['TotalClaims'] / df['HasClaim'].replace({0: 1})
    df['ClaimSeverity'] = df['ClaimSeverity'].fillna(0)
    df['Margin'] = df['TotalPremium'] - df['TotalClaims']
    return df

def check_group_balance(df, feature_cols, group_col, control, test):
    """Check balance between control and test groups on key features."""
    control_data = df[df[group_col] == control]
    test_data = df[df[group_col] == test]
    balance_report = {
        'Group': f"Control: {control}, Test: {test}",
        'Mean_RegistrationYear': f"Control {control_data['RegistrationYear'].mean():.2f}, Test {test_data['RegistrationYear'].mean():.2f}",
        'VehicleType_Dist_Control': control_data['VehicleType'].value_counts(normalize=True).to_dict(),
        'VehicleType_Dist_Test': test_data['VehicleType'].value_counts(normalize=True).to_dict(),
        'PlanType_Dist_Control': control_data['PlanType'].value_counts(normalize=True).to_dict() if 'PlanType' in df.columns else {}
    }
    return balance_report

def test_hypothesis(group_var, df, metric, control, test, alpha=0.05):
    """Perform statistical test based on metric type with data validation."""
    if df.empty or df[group_var].isnull().all() or df[metric].isnull().all():
        return {'Test': 'N/A', 'Statistic': np.nan, 'p-value': np.nan, 'Decision': 'No data for test'}
    if metric == 'ClaimFrequency':
        freq_table = pd.crosstab(df[group_var], df['HasClaim'])
        if freq_table.size == 0:
            return {'Test': 'Chi-squared', 'Statistic': np.nan, 'p-value': np.nan, 'Decision': 'No data for contingency table'}
        chi2_stat, p_val, dof, ex = stats.chi2_contingency(freq_table)
        test_type = 'Chi-squared'
        stat = chi2_stat
    else:
        control_data = df[df[group_var] == control][metric].dropna()
        test_data = df[df[group_var] == test][metric].dropna()
        if len(control_data) == 0 or len(test_data) == 0:
            return {'Test': 't-test', 'Statistic': np.nan, 'p-value': np.nan, 'Decision': 'Insufficient data for t-test'}
        t_stat, p_val = stats.ttest_ind(control_data, test_data, equal_var=False)
        test_type = 't-test'
        stat = t_stat
    decision = 'Reject H0' if p_val < alpha else 'Fail to reject H0'
    return {'Test': test_type, 'Statistic': stat, 'p-value': p_val, 'Decision': decision}

def visualize_results(df, group_var, metric, control, test):
    """Generate and display visualization in notebook."""
    plt.figure(figsize=(10, 6))
    if metric == 'ClaimFrequency':
        sns.barplot(x=group_var, y='HasClaim', data=df, errorbar=None)
        plt.title(f'{metric} by {group_var} ({control} vs {test})')
        plt.ylabel('Proportion of Policies with Claims')
    else:
        sns.boxplot(x=group_var, y=metric, data=df)
        plt.title(f'{metric} Distribution by {group_var} ({control} vs {test})')
    plt.xticks(rotation=45)
    plt.show()  # Display the plot in the notebook

def generate_business_recommendations(df, results_df):
    """Generate business recommendations based on rejected hypotheses."""
    recommendations = []
    for idx, row in results_df.iterrows():
        if row['Decision'] == 'Reject H0':
            if 'provinces' in row['Hypothesis']:
                prov_diff = df[df['Province'] == row['Control']]['LossRatio'].mean() - df[df['Province'] == row['Test']]['LossRatio'].mean()
                recommendations.append(f"- {row['Hypothesis']}: We reject H0 (p = {row['p-value']:.4f}). {row['Control']} shows a {prov_diff:.2%} higher loss ratio than {row['Test']}, suggesting a regional premium adjustment.")
            elif 'zip codes' in row['Hypothesis'] and 'Frequency' in row['Hypothesis']:
                freq_diff = df[df['ZipGroup'] == row['Test']]['HasClaim'].mean() - df[df['ZipGroup'] == row['Control']]['HasClaim'].mean()
                recommendations.append(f"- {row['Hypothesis']}: We reject H0 (p = {row['p-value']:.4f}). {row['Test']} zip group shows a {freq_diff:.2%} higher claim frequency, indicating zip-code-based risk segmentation.")
            elif 'zip codes' in row['Hypothesis'] and 'Margin' in row['Hypothesis']:
                margin_diff = df[df['ZipGroup'] == row['Test']]['Margin'].mean() - df[df['ZipGroup'] == row['Control']]['Margin'].mean()
                recommendations.append(f"- {row['Hypothesis']}: We reject H0 (p = {row['p-value']:.4f}). {row['Test']} zip group shows a ${margin_diff:.2f} higher margin, suggesting targeted pricing.")
            elif 'genders' in row['Hypothesis']:
                gender_diff = df[df['Gender'] == row['Control']]['LossRatio'].mean() - df[df['Gender'] == row['Test']]['LossRatio'].mean()
                recommendations.append(f"- {row['Hypothesis']}: We reject H0 (p = {row['p-value']:.4f}). {row['Control']} shows a {gender_diff:.2%} higher loss ratio than {row['Test']}, suggesting gender-based risk adjustments.")
    return recommendations

if __name__ == "__main__":
    df = pd.read_csv('data/processed/cleaned_data.csv')
    df = prepare_data(df)
    df['ZipGroup'] = pd.qcut(df['PostalCode'].rank(method='first'), 2, labels=['Low', 'High'])
    results = []
    prov_result = test_hypothesis('Province', df[df['Province'].isin(['Gauteng', 'Western Cape'])], 'ClaimFrequency', 'Gauteng', 'Western Cape')
    results.append({'Hypothesis': 'No risk differences across provinces (Frequency)', 'Control': 'Gauteng', 'Test': 'Western Cape', **prov_result})
    print(results)