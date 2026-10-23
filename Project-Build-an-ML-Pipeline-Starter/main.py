import json
import os
os.environ["GIT_CEILING_DIRECTORIES"] = os.path.dirname(os.path.abspath(__file__))
import mlflow
import tempfile
import wandb
import hydra
from omegaconf import DictConfig

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
]

@hydra.main(version_base=None, config_name='config', config_path='.')
def go(config: DictConfig):
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    with tempfile.TemporaryDirectory() as tmp_dir:
        if "download" in active_steps:
            _ = mlflow.run(
                os.path.join(config['main']['components_repository'], "get_data"),
                "main",
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
                "src/basic_cleaning",
                "main",
                env_manager="local",
                parameters={
                    "input_artifact": "sample.csv:latest",
                    "output_artifact": "clean_sample.csv",
                    "output_type": "dataset",
                    "output_description": "Cleaned Airbnb dataset",
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"],
                },
            )

        if "data_check" in active_steps:
            _ = mlflow.run(
                "src/data_check",
                "main",
                env_manager="local",
                parameters={
                    "csv": "clean_sample.csv:latest",
                    "ref": "sample.csv:latest",
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"],
                },
            )

        if "data_split" in active_steps:
        _ = mlflow.run(
            f"{config['main']['components_repository']}/train_val_test_split",
            'main',
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
            f"{config['main']['components_repository']}/train_random_forest",
            'main',
            env_manager="local",
            env_dict = {
                **os.environ,
                "WANDB_API_KEY": os.environment.get("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")
                "WANDB_PROJECT": config["main"]["project_name"],
                "WANDB_RUN_GROUP": config["main"]["experiment_name"],
                "WANDB_MODE": "online",
            },
            parameters={
                "trainval_artifact": "trainval_data.csv:latest",
                "val_size": config['modeling']['val_size'] 
                "random_seed": config['modeling']['random_seed'],
                "stratify_by": config['modeling']['stratify_by'],
                "rf_config": rf_config,
                "output_artifact": "random_forest_export",
            }
        )

        if "test_regression_model" in active_steps:
            pass

if __name__ == '__main__':
    go()
