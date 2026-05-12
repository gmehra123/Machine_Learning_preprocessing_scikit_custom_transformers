"""ML Pipeline configuration: hyperparameters, column definitions, drift thresholds."""
TARGET_COL = "Churn"
NUMERIC_FEATURES = ["Age", "Tenure", "Usage Frequency", "Support Calls", "Payment Delay", "Total Spend", "Last Interaction"]
CATEGORICAL_FEATURES = ["Gender", "Subscription Type", "Contract Length"]
ID_COL = "CustomerID"
IMPUTER_GROUP_COL = "Gender"
TEST_SIZE = 0.15
RANDOM_STATE = 42
DRIFT_PSI_WARNING = 0.2
DRIFT_PSI_CRITICAL = 0.25
DRIFT_CHI2_THRESHOLD = 0.05
MODEL_TYPE = "LogisticRegression"
MODEL_PARAMS = {"max_iter": 1000, "random_state": 42}
