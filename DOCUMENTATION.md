# Machine Learning Preprocessing Documentation
Generated: 2026-05-11 23:10:49
## Custom Transformers
### GroupedImputer
**Class:** `GroupedImputer`

**Description:** Imputes missing values for a single column using grouped mean strategy. Fills NaN values in the specified aggregation column (agg_col) with the mean of values belonging to the same group defined by group_col. Falls back to global mean if a group is not found in training data.

**Parameters:**
- `group_col`: Column name to group by (e.g., 'Gender')
- `agg_col`: Column name with missing values to impute (e.g., 'Usage Frequency')

**Methods:** fit, transform
### GroupedImputerFull
**Class:** `GroupedImputerFull`

**Description:** Automatically imputes missing values for ALL columns with NaN using grouped mean strategy. Automatically detects columns containing missing values and imputes each using group-based means from a specified grouping column. More versatile than GroupedImputer as it handles multiple columns at once.

**Parameters:**
- `group_col`: Column name to group by (e.g., 'Gender')

**Methods:** fit, transform
### OutlierClipper
**Class:** `OutlierClipper`

**Description:** Clips outlier values to specified quantile boundaries. Useful for handling extreme values in numerical features by capping them at the configured percentiles (default: 25th and 75th). Helps reduce the impact of outliers on machine learning models.

**Parameters:**
- `lower_quantile`: Lower quantile threshold (default=0.25)
- `upper_quantile`: Upper quantile threshold (default=0.75)

**Methods:** fit, transform

## Data Pipeline Steps
1. **Data Loading**: Load customer churn dataset from CSV file
2. **Initial Cleaning**: Remove rows with missing values using dropna()
3. **Missing Value Simulation**: Add random NaN values to 'Payment Delay' and 'Usage Frequency' columns to simulate real-world missing data
4. **Train-Test Split**: Split data into train (85%) and test (15%) with stratification on target variable
5. **Grouped Imputation**: Impute missing values using Gender-grouped means with fallback to global mean
6. **Feature Clipping**: Clip outlier values using quantile boundaries (Q1=0.25, Q3=0.75)

## Usage Example

```python
from sklearn.model_selection import train_test_split
import pandas as pd

# Load data
data = pd.read_csv('customer_churn_dataset-training-master.csv')
data.dropna(inplace=True)

# Add random missing values (simulation)
import numpy as np
def add_random_nan_values(df, cols, frac=0.02):
    for col in cols:
        row_idx = df.sample(frac=frac).index
        df.loc[row_idx, col] = np.nan
add_random_nan_values(data, ['Payment Delay', 'Usage Frequency'])

# Split features and target
X = data.drop(columns='Churn')
y = data['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=12)

# Impute missing values using grouped means
from documentation_agent import GroupedImputerFull
imputer = GroupedImputerFull(group_col='Gender')
X_train_imputed = imputer.fit_transform(X_train)
```
