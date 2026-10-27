import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
from ml.model import train_model, compute_model_metrics

@pytest.fixture
def sample_data():
    data = {
        "age": [39, 50, 38, 53, 28],
        "workclass": ["State-gov", "Self-emp-not-inc", "Private", "Private", "Private"],
        "fnlwgt": [77516, 83311, 215646, 234721, 338409],
        "education": ["Bachelors", "Bachelors", "HS-grad", "11th", "Bachelors"],
        "education-num": [13, 13, 9, 7, 13],
        "marital-status": ["Never-married", "Married-civ-spouse", "Divorced", "Married-civ-spouse", "Married-civ-spouse"],
        "occupation": ["Adm-clerical", "Exec-managerial", "Handlers-cleaners", "Handlers-cleaners", "Prof-specialty"],
        "relationship": ["Not-in-family", "Husband", "Not-in-family", "Husband", "Wife"],
        "race": ["White", "White", "White", "Black", "Black"],
        "sex": ["Male", "Male", "Male", "Male", "Female"],
        "capital-gain": [2174, 0, 0, 0, 0],
        "capital-loss": [0, 0, 0, 0, 0],
        "hours-per-week": [40, 13, 40, 40, 40],
        "native-country": ["United-States", "United-States", "United-States", "United-States", "Cuba"],
        "income": ["<=50K", "<=50K", "<=50K", "<=50K", "<=50K"]
    }
    return pd.DataFrame(data)

def test_process_data(sample_data):
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    X_train, y_train, encoder, lb = process_data(
        sample_data, categorical_features=cat_features, label="income", training=True
    )
    assert hasattr(X_train, "shape")
    assert len(X_train) == len(sample_data)
    assert len(y_train) == len(sample_data)

def test_train_model(sample_data):
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    X_train, y_train, encoder, lb = process_data(
        sample_data, categorical_features=cat_features, label="income", training=True
    )
    model = train_model(X_train, y_train, random_state=42)
    assert isinstance(model, RandomForestClassifier)

def test_compute_model_metrics():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= fbeta <= 1.0
