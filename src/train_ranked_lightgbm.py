from src.config import PROJECT_ROOT
from src.factor_library import (
    LABEL_NAME,
    PRICE_VOLUME_FEATURE_NAMES,
    RANK_LABEL_NAME,
)
from src.train_lightgbm import (
    train_lightgbm_experiment,
)


def main() -> None:
    dataset_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "ranked_multifactor_dataset.parquet"
    )

    train_lightgbm_experiment(
        dataset_path=dataset_path,
        feature_names=PRICE_VOLUME_FEATURE_NAMES,
        target_name=RANK_LABEL_NAME,
        model_name="lightgbm_ranked",
        retained_columns=[
            LABEL_NAME,
            RANK_LABEL_NAME,
        ],
    )


if __name__ == "__main__":
    main()
