"""Training entry point: load data, detect drift, train, evaluate, save model."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loaders import load_csv, split_data
from src.models.train import train_pipeline, evaluate_pipeline
from src.models.registry import save_model
from src.utils.config import TARGET_COL
from src.utils.logging import get_logger, log_metrics, log_pipeline_stage

logger = get_logger(__name__)


def main():
    log_pipeline_stage(logger, "LOAD DATA")
    df = load_csv("customer_churn_dataset-training-master.csv")
    df.dropna(subset=[TARGET_COL], inplace=True)
    logger.info(f"Data shape after dropping NaN target: {df.shape}")

    X_train, X_test, y_train, y_test = split_data(df, TARGET_COL, test_size=0.15, random_state=42)

    log_pipeline_stage(logger, "TRAIN PIPELINE")
    pipeline = train_pipeline(X_train, y_train)

    log_pipeline_stage(logger, "EVALUATE ON HELD-OUT TEST SET")
    test_metrics = evaluate_pipeline(pipeline, X_test, y_test)
    log_metrics(logger, test_metrics, stage="TEST")

    log_pipeline_stage(logger, "SAVE MODEL")
    model_path = save_model(pipeline, name="churn_model", version=1)
    logger.info(f"Training complete. Model saved to: {model_path}")

    print()
    logger.info("=" * 50)
    logger.info("FINAL TEST METRICS")
    logger.info("=" * 50)
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()