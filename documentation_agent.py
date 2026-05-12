from datetime import datetime


class DocumentationAgent:
    def __init__(self, notebook_path=None):
        self.notebook_path = notebook_path
        self.transformers = []
        self.pipeline_steps = []

    def add_transformer(self, name, transformer_class, description, parameters, fit_method=True, transform_method=True):
        self.transformers.append({
            "name": name,
            "class": transformer_class,
            "description": description,
            "parameters": parameters,
            "has_fit": fit_method,
            "has_transform": transform_method
        })

    def add_pipeline_step(self, step_name, description):
        self.pipeline_steps.append({"name": step_name, "description": description})

    def generate_markdown(self):
        md = []
        md.append(f"# Machine Learning Preprocessing Documentation")
        md.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        md.append("## Custom Transformers\n")
        for t in self.transformers:
            md.append(f"### {t['name']}\n")
            md.append(f"**Class:** `{t['class']}`\n\n")
            md.append(f"**Description:** {t['description']}\n\n")
            md.append(f"**Parameters:**\n")
            for param, desc in t['parameters'].items():
                md.append(f"- `{param}`: {desc}\n")
            md.append(f"\n**Methods:** ")
            methods = []
            if t['has_fit']:
                methods.append("fit")
            if t['has_transform']:
                methods.append("transform")
            md.append(", ".join(methods) + "\n")

        if self.pipeline_steps:
            md.append("\n## Data Pipeline Steps\n")
            for i, step in enumerate(self.pipeline_steps, 1):
                md.append(f"{i}. **{step['name']}**: {step['description']}\n")

        md.append("\n## Usage Example\n\n")
        md.append("```python\n")
        md.append("from sklearn.model_selection import train_test_split\n")
        md.append("import pandas as pd\n\n")
        md.append("# Load data\n")
        md.append("data = pd.read_csv('customer_churn_dataset-training-master.csv')\n")
        md.append("data.dropna(inplace=True)\n\n")
        md.append("# Add random missing values (simulation)\n")
        md.append("import numpy as np\n")
        md.append("def add_random_nan_values(df, cols, frac=0.02):\n")
        md.append("    for col in cols:\n")
        md.append("        row_idx = df.sample(frac=frac).index\n")
        md.append("        df.loc[row_idx, col] = np.nan\n")
        md.append("add_random_nan_values(data, ['Payment Delay', 'Usage Frequency'])\n\n")
        md.append("# Split features and target\n")
        md.append("X = data.drop(columns='Churn')\n")
        md.append("y = data['Churn']\n\n")
        md.append("# Train-test split\n")
        md.append("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=12)\n\n")
        md.append("# Impute missing values using grouped means\n")
        md.append("from documentation_agent import GroupedImputerFull\n")
        md.append("imputer = GroupedImputerFull(group_col='Gender')\n")
        md.append("X_train_imputed = imputer.fit_transform(X_train)\n")
        md.append("```\n")

        return "".join(md)

    def save_documentation(self, output_path="DOCUMENTATION.md"):
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self.generate_markdown())
        print(f"Documentation saved to {output_path}")


if __name__ == "__main__":
    agent = DocumentationAgent()

    agent.add_transformer(
        name="GroupedImputer",
        transformer_class="GroupedImputer",
        description="Imputes missing values for a single column using grouped mean strategy. Fills NaN values in the specified aggregation column (agg_col) with the mean of values belonging to the same group defined by group_col. Falls back to global mean if a group is not found in training data.",
        parameters={
            "group_col": "Column name to group by (e.g., 'Gender')",
            "agg_col": "Column name with missing values to impute (e.g., 'Usage Frequency')"
        }
    )

    agent.add_transformer(
        name="GroupedImputerFull",
        transformer_class="GroupedImputerFull",
        description="Automatically imputes missing values for ALL columns with NaN using grouped mean strategy. Automatically detects columns containing missing values and imputes each using group-based means from a specified grouping column. More versatile than GroupedImputer as it handles multiple columns at once.",
        parameters={
            "group_col": "Column name to group by (e.g., 'Gender')"
        }
    )

    agent.add_transformer(
        name="OutlierClipper",
        transformer_class="OutlierClipper",
        description="Clips outlier values to specified quantile boundaries. Useful for handling extreme values in numerical features by capping them at the configured percentiles (default: 25th and 75th). Helps reduce the impact of outliers on machine learning models.",
        parameters={
            "lower_quantile": "Lower quantile threshold (default=0.25)",
            "upper_quantile": "Upper quantile threshold (default=0.75)"
        }
    )

    agent.add_pipeline_step("Data Loading", "Load customer churn dataset from CSV file")
    agent.add_pipeline_step("Initial Cleaning", "Remove rows with missing values using dropna()")
    agent.add_pipeline_step("Missing Value Simulation", "Add random NaN values to 'Payment Delay' and 'Usage Frequency' columns to simulate real-world missing data")
    agent.add_pipeline_step("Train-Test Split", "Split data into train (85%) and test (15%) with stratification on target variable")
    agent.add_pipeline_step("Grouped Imputation", "Impute missing values using Gender-grouped means with fallback to global mean")
    agent.add_pipeline_step("Feature Clipping", "Clip outlier values using quantile boundaries (Q1=0.25, Q3=0.75)")

    agent.save_documentation("DOCUMENTATION.md")