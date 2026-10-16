"""Train the 15-factor LightGBM ablation model."""

from src.config import PROJECT_ROOT
from src.factor_library import (
    ENHANCED_FEATURE_NAMES,
    LABEL_NAME,
)
from src.train_lightgbm import train_lightgbm_experiment


def main() -> None:
    dataset_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "enhanced_multifactor_dataset.parquet"
    )

    train_lightgbm_experiment(
        dataset_path=dataset_path,
        feature_names=ENHANCED_FEATURE_NAMES,
        target_name=LABEL_NAME,
        model_name="lightgbm_enhanced",
        retained_columns=[
            LABEL_NAME,
        ],
    )


if __name__ == "__main__":
    main()
