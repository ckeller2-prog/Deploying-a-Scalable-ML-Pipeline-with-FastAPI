import os
# Prevent git from walking up to parent directories and triggering git repo checks
os.environ["GIT_CEILING_DIRECTORIES"] = "/home"

import json
import mlflow
import tempfile
import pathlib
import wandb
import hydra
from omegaconf import DictConfig

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    "test_regression_model"
]

@hydra.main(version_base=None, config_name='config', config_path='.')
def go(config: DictConfig):

    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    cwd = pathlib.Path(hydra.utils.get_original_cwd())

    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            _ = mlflow.run(
                str(cwd / "components" / "get_data"),
                "main",
                version=None,
                env_manager="local",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )

        if "basic_cleaning" in active_steps:
            _ = mlflow.run(
                str(cwd / "src" / "basic_cleaning"),
                "main",
                version=None,
                env_manager="local",
                parameters={
                    "input_artifact": "sample.csv:latest",
                    "output_artifact": "clean_sample.csv",
                    "output_type": "dataset",
                    "output_description": "Cleaned sample of Airbnb data",
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"],
                },
            )
                     
        if "data_check" in active_steps:
            _ = mlflow.run(
                str(cwd / "src" / "data_check"),
                "main",
                version=None,
                env_manager="local",
                parameters={
                    "csv": "clean_sample.csv:latest",
                    "ref": "clean_sample.csv:latest",
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"],
                },
            )

        if "data_split" in active_steps:
            _ = mlflow.run(
                str(cwd / "components" / "train_val_test_split"),
                'main',
                version=None,
                env_manager="local",
                parameters={
                    "input": "clean_sample.csv:latest",
                    "test_size": config['modeling']['test_size'],
                    "random_seed": config['modeling']['random_seed'],
                    "stratify_by": config['modeling']['stratify_by']
                }
            )

        if "train_random_forest" in active_steps:
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)
            
            _ = mlflow.run(
                str(cwd / "components" / "train_random_forest"),
                'main',
                version=None,                
                env_manager="local",
                parameters={
                    "trainval_artifact": "trainval_data.csv:latest",
                    "val_size": config['modeling']['val_size'],
                    "random_seed": config['modeling']['random_seed'],
                    "stratify_by": config['modeling']['stratify_by'],
                    "rf_config": rf_config,
                    "max_tfidf_features": config['modeling']['max_tfidf_features'],
                    "output_artifact": "random_forest_export"
                }
            )

        if "test_regression_model" in active_steps:
            _ = mlflow.run(
                str(cwd / "components" / "test_regression_model"),
                'main',
                version=None,
                env_manager="local",
                parameters={
                    "mlflow_model": "random_forest_export:latest",
                    "test_dataset": "test_data.csv:latest"
                }
            )
            
if __name__ == "__main__":
    go()
