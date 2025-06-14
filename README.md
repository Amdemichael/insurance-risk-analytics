# Insurance Risk Analytics Project

This repository contains the exploratory data analysis (EDA) and data version control (DVC) implementation for the Insurance Risk Analytics project. The project analyzes an insurance dataset to derive insights and establish a reproducible data pipeline.

## Table of Contents
- [Overview](#overview)
- [Task 1: Exploratory Data Analysis (EDA)](#task-1-exploratory-data-analysis-eda)
- [Task 2: Data Version Control with DVC](#task-2-data-version-control-with-dvc)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview
This project is part of a data science workflow to analyze an insurance dataset (`MachineLearningRating_v3.txt`) containing policy and claim information. Task 1 focuses on performing EDA to uncover patterns and relationships, while Task 2 implements DVC to manage data versions and storage. The work is conducted using Python, Jupyter Notebooks, and Git for version control, with DVC for data versioning.

## Task 1: Exploratory Data Analysis (EDA)
### Objectives
- Perform initial data exploration and cleaning.
- Visualize distributions, correlations, and trends in the insurance data.
- Identify outliers and assess data quality.

### Methods
- Data is loaded and cleaned using a custom `eda.py` module.
- Visualizations (e.g., histograms, scatter plots, heatmaps) are generated using Matplotlib and Seaborn.
- The analysis is executed via the `exploratory_analysis.ipynb` Jupyter Notebook.

### Outputs
- Cleaned dataset: `data/processed/cleaned_data.csv`.
- Visualizations saved in `data/plots/` (e.g., `premium_vs_claims_scatter.png`).
- Inline plots displayed in the notebook for interactive exploration.

## Task 2: Data Version Control with DVC
### Objectives
- Set up DVC to version-control the cleaned dataset.
- Configure a local remote storage for data files.
- Ensure reproducibility of the data pipeline.

### Methods
- DVC repository initialized with `dvc init`.
- Remote storage configured at `D:\Tutorials\KAIM\Weeks\Week 3\dvc_storage` using `dvc remote add -d localstorage`.
- The cleaned dataset (`data/processed/cleaned_data.csv`) is tracked with `dvc add` and pushed to the remote with `dvc push`.

### Outputs
- DVC metadata file: `data/processed/cleaned_data.csv.dvc`.
- Data stored in the remote `dvc_storage` directory.
- Git commits include `.dvc/config` and `.dvc` files.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/insurance-risk-analytics.git
   cd insurance-risk-analytics
   ```

- Install dependencies:
```bash
    pip install -r requirements.txt
```

- Install DVC:

```bash
    pip install dvc
```

- Initialize DVC (if not already done):
```bash
    dvc init
```

- Set up the remote storage (adjust path as needed):
```bash
    dvc remote add -d localstorage D:\Tutorials\KAIM\Weeks\Week 3\dvc_storage
```

## Usage
1. Launch Jupyter Notebook:

- Open notebooks/exploratory_analysis.ipynb and run all cells to perform EDA and generate plots.
- Track and push data with DVC:
```bash
    dvc add data\processed\cleaned_data.csv
    dvc push
```

- Commit changes to Git:
```bash
    git add .
    git commit -m "Update EDA and DVC"
    git push origin task-1
```


#### **8. File Structure**
- Describe the project layout.
```markdown

## File Structure
INSURANCE-RISK-ANALYTICS/
├── .dvc
├── .github/
│   └── workflows/
├── pytest.yml
├── data/
│   ├── plots/
│   └── processed/
│   └── MachineLearningRating_v3.txt
├── docs/
├── dvc_storage/
├── insurance/
├── notebooks/
├── src/
├── tests/
├── .gitignore
├── README.md
└── requirements.txt

insurance-risk-analytics/
data/: Raw and processed data files.
MachineLearningRating_v3.txt: Input dataset.

processed/: Directory for cleaned data (e.g., cleaned_data.csv).

plots/: Directory for saved visualizations.

notebooks/: Jupyter notebooks for analysis.
exploratory_analysis.ipynb: EDA notebook.

src/: Source code.
eda.py: EDA functions and visualizations.

.dvc/: DVC configuration and cache.

.gitignore: Git ignore file.

README.md: This file.

requirements.txt: Python dependencies.

