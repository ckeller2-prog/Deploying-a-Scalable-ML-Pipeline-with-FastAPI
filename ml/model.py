import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data

def train_model(X_train, y_train, **kwargs):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array or pd.DataFrame
        Training data.
    y_train : np.array or pd.Series
        Labels.
    **kwargs : dict
        Keyword arguments for the classifier.

    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier(**kwargs)
    model.fit(X_train, y_train)
    return model

def inference(model, X):
    """
    Run model inferences and return the predictions.

    Inputs
    ------
    model : trained machine learning model
        Trained machine learning model.
    X : np.array or pd.DataFrame
        Data used for prediction.

    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)

def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and f1.

    Inputs
    ------
    y : np.array or pd.Series
        Known labels, binarized.
    preds : np.array or pd.Series
        Predicted labels, binarized.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta

def save_model(model, path):
    """
    Saves a model or any categorical encoders to disk.

    Inputs
    ------
    model : object
        Model or encoder to save.
    path : str
        File path to save the object.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def load_model(path):
    """
    Loads a model or any categorical encoders from disk.

    Inputs
    ------
    path : str
        File path of the saved object.

    Returns
    -------
    object
        Loaded model or encoder.
    """
    return joblib.load(path)

def performance_on_categorical_slice(df, feature, target_col, model, encoder=None, lb=None):
    """
    Computes performance metrics on slices of the data for a given categorical feature.

    Inputs
    ------
    df : pd.DataFrame
        DataFrame containing features and target.
    feature : str
        Categorical feature to slice by.
    target_col : str
        Name of the target column.
    model : trained model
        Trained machine learning model.
    encoder : fitted encoder, optional
        Fitted OneHotEncoder or preprocessor.
    lb : fitted LabelBinarizer, optional
        Fitted LabelBinarizer.

    Returns
    -------
    slice_metrics : dict
        Dictionary containing metrics for each unique value of the feature.
    """
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    slice_metrics = {}
    for val in df[feature].unique():
        df_slice = df[df[feature] == val]
        if len(df_slice) == 0:
            continue

        X_slice, y_slice, _, _ = process_data(
            df_slice, 
            categorical_features=cat_features, 
            label=target_col, 
            training=False, 
            encoder=encoder, 
            lb=lb
        )
        
        preds = inference(model, X_slice)
        precision, recall, fbeta = compute_model_metrics(y_slice, preds)
        
        slice_metrics[val] = {
            "precision": precision,
            "recall": recall,
            "fbeta": fbeta,
            "count": len(df_slice)
        }
    return slice_metrics
