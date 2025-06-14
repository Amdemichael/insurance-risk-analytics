import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, probplot

# References:
# - Pandas Documentation: https://pandas.pydata.org/docs/
# - Seaborn Visualization Guide: https://seaborn.pydata.org/tutorial.html
# - Statistics for Data Science: "Practical Statistics for Data Scientists" by Peter Bruce

def get_project_root(data_file_path):
    """Determine the project root based on the data file location."""
    if os.path.isabs(data_file_path):
        return os.path.dirname(os.path.dirname(data_file_path))  # Go up two levels from data file
    else:
        return os.path.abspath(os.path.join(os.getcwd(), '..'))  # Relative to current working dir

# Initialize paths based on data file location
DATA_FILE_PATH = 'data/MachineLearningRating_v3.txt'  # Default relative path
PROJECT_ROOT = get_project_root(DATA_FILE_PATH)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'data', 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_data(file_path=DATA_FILE_PATH):
    """Load data from a pipe-separated text file and handle basic data types."""
    if not os.path.isabs(file_path):
        file_path = os.path.join(DATA_DIR, os.path.basename(file_path))
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")
    df = pd.read_csv(file_path, sep='|', encoding='utf-8')
    df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
    for col in ['TotalPremium', 'TotalClaims', 'SumInsured', 'CalculatedPremiumPerTerm', 
                'CustomValueEstimate', 'CapitalOutstanding', 'kilowatts', 'cubiccapacity', 
                'RegistrationYear']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def summarize_data(df):
    """Summarize data with descriptive statistics and calculate LossRatio."""
    print("Descriptive Statistics for Numerical Features:")
    print(df[['TotalPremium', 'TotalClaims', 'SumInsured', 'CalculatedPremiumPerTerm', 
              'CustomValueEstimate']].describe())
    df['LossRatio'] = df['TotalClaims'] / df['TotalPremium'].replace(0, 1e-10)
    overall_loss_ratio = df['LossRatio'].mean()
    print(f"Overall Loss Ratio for the portfolio: {overall_loss_ratio:.2f}")
    print("\nData Types of Columns:")
    print(df.dtypes)
    return df

def assess_data_quality(df):
    """Assess and handle missing values."""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    print("\nMissing Values (Count and Percentage):")
    print(pd.DataFrame({'Missing': missing, 'Percentage': missing_pct}))
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def plot_distributions(df):
    """Plot histograms for numerical columns and bar charts for categorical columns."""
    numerical_cols = ['TotalClaims', 'TotalPremium', 'CustomValueEstimate']
    for col in numerical_cols:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col], bins=50, kde=True)
        plt.title(f'Distribution of {col}')
        plt.savefig(os.path.join(PLOTS_DIR, f'{col}_hist.png'))
        plt.show()  # Display inline in notebook
        plt.close()
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Province', data=df)
    plt.title('Policy Count by Province')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(PLOTS_DIR, 'province_bar.png'))
    plt.show()  # Display inline in notebook
    plt.close()

def bivariate_analysis(df):
    """Explore correlations and relationships using scatter plots and correlation matrices."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='TotalPremium', y='TotalClaims', hue='PostalCode', data=df)
    plt.title('Total Premium vs Total Claims by PostalCode')
    plt.savefig(os.path.join(PLOTS_DIR, 'premium_vs_claims_scatter.png'))
    plt.show()  # Display inline in notebook
    plt.close()
    corr = df[['TotalPremium', 'TotalClaims', 'SumInsured', 'CalculatedPremiumPerTerm']].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig(os.path.join(PLOTS_DIR, 'correlation_matrix.png'))
    plt.show()  # Display inline in notebook
    plt.close()

def detect_outliers(df):
    """Detect outliers using box plots for numerical data."""
    numerical_cols = ['TotalClaims', 'TotalPremium', 'CustomValueEstimate']
    for col in numerical_cols:
        plt.figure(figsize=(8, 6))
        sns.boxplot(y=col, data=df)
        plt.title(f'Box Plot of {col}')
        plt.savefig(os.path.join(PLOTS_DIR, f'{col}_box.png'))
        plt.show()  # Display inline in notebook
        plt.close()
    return df

def trends_over_geography(df):
    """Compare trends over geography (e.g., Province, Premium, Auto Make)."""
    province_loss = df.groupby('Province')['LossRatio'].mean().sort_values()
    plt.figure(figsize=(8, 6))
    province_loss.plot(kind='bar')
    plt.title('Average Loss Ratio by Province')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(PLOTS_DIR, 'loss_ratio_province.png'))
    plt.show()  # Display inline in notebook
    plt.close()
    make_claims = df.groupby('make')['TotalClaims'].sum().nlargest(10)
    plt.figure(figsize=(10, 6))
    make_claims.plot(kind='bar')
    plt.title('Top 10 Vehicle Makes by Total Claims')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(PLOTS_DIR, 'top_makes_claims.png'))
    plt.show()  # Display inline in notebook
    plt.close()

def group_statistics(df):
    """Calculate LossRatio by VehicleType and Gender."""
    print("\nLossRatio by VehicleType:")
    print(df.groupby('VehicleType')['LossRatio'].mean().sort_values())
    print("\nLossRatio by Gender:")
    print(df.groupby('Gender')['LossRatio'].mean().sort_values())

def statistical_tests(df):
    """Perform normality test and probability plot for TotalClaims."""
    stat, p = shapiro(df['TotalClaims'].dropna())
    print(f"\nShapiro-Wilk Test for TotalClaims: Statistic={stat:.3f}, p-value={p:.3f}")
    if p < 0.05:
        print("Reject null hypothesis: TotalClaims is not normally distributed.")
    plt.figure(figsize=(8, 6))
    probplot(df['TotalClaims'].dropna(), dist="norm", plot=plt)
    plt.title('Q-Q Plot for TotalClaims')
    plt.savefig(os.path.join(PLOTS_DIR, 'qq_plot_totalclaims.png'))
    plt.show()  # Display inline in notebook
    plt.close()

def creative_visualizations(df):
    """Generate 3 creative and insightful plots."""
    province_pivot = df.pivot_table(values='LossRatio', index='Province', aggfunc='mean')
    plt.figure(figsize=(10, 8))
    sns.heatmap(province_pivot, annot=True, cmap='YlOrRd')
    plt.title('Loss Ratio Heatmap by Province')
    plt.savefig(os.path.join(PLOTS_DIR, 'loss_ratio_heatmap.png'))
    plt.show()  # Display inline in notebook
    plt.close()
    monthly_claims = df.groupby(df['TransactionMonth'].dt.to_period('M'))['TotalClaims'].sum()
    plt.figure(figsize=(10, 6))
    monthly_claims.plot()
    plt.title('Total Claims Over Time (Feb 2014 - Aug 2015)')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(PLOTS_DIR, 'claims_over_time.png'))
    plt.show()  # Display inline in notebook
    plt.close()
    vehicle_premium = df.groupby('VehicleType')['CalculatedPremiumPerTerm'].mean().sort_values()
    plt.figure(figsize=(10, 6))
    vehicle_premium.plot(kind='bar')
    plt.title('Average Premium by Vehicle Type')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(PLOTS_DIR, 'premium_by_vehicle_type.png'))
    plt.show()  # Display inline in notebook
    plt.close()

if __name__ == "__main__":
    df = load_data()
    df = summarize_data(df)
    df = assess_data_quality(df)
    plot_distributions(df)
    bivariate_analysis(df)
    df = detect_outliers(df)
    trends_over_geography(df)
    advanced_pairplot(df)
    group_statistics(df)
    statistical_tests(df)
    creative_visualizations(df)
    df.to_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'cleaned_data.csv'), index=False)
    print(f"EDA complete. Visualizations saved in {PLOTS_DIR}. Cleaned data saved as cleaned_data.csv.")