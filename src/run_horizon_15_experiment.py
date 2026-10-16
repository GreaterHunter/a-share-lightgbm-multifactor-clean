"""Build and train the 15-day ranked-enhanced annual rolling experiment."""

from src.run_horizon_10_experiment import run_horizon_experiment


if __name__ == "__main__":
    run_horizon_experiment(15)
