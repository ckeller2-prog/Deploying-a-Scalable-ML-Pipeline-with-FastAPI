import argparse
import os
import shutil
import json
import pandas as pd
import wandb
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

def get_inference_pipeline(rf_config, max_tfidf_features, X_columns):
    numeric_features = rf_config.get("numeric_features", [])
    categorical_features = rf_config.get("categorical_features", [])
    
    if not numeric_features:
        numeric_features = list(X_columns.select_dtypes(include=["number"]).columns)
    if not categorical_features:
        categorical_features = list(X_columns.select_dtypes(exclude=["number"]).columns)
        
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical_features)
        ],
        remainder="drop"
    )
    
    model = RandomForestRegressor(
        n_estimators=rf_config.get("n_estimators", 100),
        max_depth=rf_config.get("max_depth", None),
        random_state=rf_config.get("random_state", 42),
        n_jobs=-1
    )
    
    sk_pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    return sk_pipe, numeric_features + categorical_features

def go(args):
    run = wandb.init(job_type="train_random_forest")
    run.config.update(args)
    artifact_local_path = run.use_artifact(args.trainval_artifact).file()
    train_df = pd.read_csv(artifact_local_path)
    
    with open(args.rf_config, 'r') as f:
        rf_config = json.load(f)
        
    target = rf_config.get("target", "price")
    if target in train_df.columns:
        y = train_df[target]
        X = train_df.drop(columns=[target])
    else:
        y = train_df.iloc[:, -1]
        X = train_df.iloc[:, :-1]
        
    drop_cols = [c for c in ["id", "name", "host_id", "host_name", "last_review"] if c in X.columns]
    if drop_cols:
        X = X.drop(columns=drop_cols)
        
    print("Minimum price:", y.min(), "Maximum price:", y.max())
    print("Preparing sklearn pipeline")
    
    sk_pipe, processed_features = get_inference_pipeline(rf_config, args.max_tfidf_features, X)
    
    print("Fitting")
    sk_pipe.fit(X, y)
    
    print("Scoring")
    preds = sk_pipe.predict(X)
    score = r2_score(y, preds)
    mae = mean_absolute_error(y, preds)
    
    print(f"Score: {score}")
    print(f"MAE: {mae}")
    print("Exporting model")
    
    if os.path.exists(args.output_artifact):
        if os.path.isdir(args.output_artifact):
            shutil.rmtree(args.output_artifact)
        else:
            os.remove(args.output_artifact)

    if os.path.exists("random_forest_dir"):
        shutil.rmtree("random_forest_dir")
    os.makedirs("random_forest_dir", exist_ok=True)

    joblib.dump(sk_pipe, os.path.join("random_forest_dir", "model.pkl"))

    mlmodel_content = {
        "artifact_path": "model",
        "flavors": {
            "python_function": {
                "loader_module": "mlflow.sklearn",
                "model_path": "model.pkl",
                "python_version": "3.10.13"
            },
            "sklearn": {
                "code_path": "",
                "model_path": "model.pkl",
                "serialization_format": "cloudpickle",
                "sklearn_version": "1.3.2"
            }
        },
        "mlflow_version": "2.15.1",
        "model_uuid": "1234567890abcdef",
    }
    with open(os.path.join("random_forest_dir", "MLmodel"), "w") as f:
        yaml.dump(mlmodel_content, f)

    with open(os.path.join("random_forest_dir", "requirements.txt"), "w") as f:
        f.write("scikit-learn\ncloudpickle\nnumpy\npandas\nskops\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainval_artifact", type=str, required=True)
    parser.add_argument("--val_size", type=float, required=True)
    parser.add_argument("--random_seed", type=int, required=True)
    parser.add_argument("--stratify_by", type=str, required=True)
    parser.add_argument("--rf_config", type=str, required=True)
    parser.add_argument("--max_tfidf_features", type=int, required=True)
    parser.add_argument("--output_artifact", type=str, required=True)
    args = parser.parse_args()
    go(args)
