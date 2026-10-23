import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

def process_data(
    X, categorical_features=[], label=None, training=True, encoder=None, lb=None
):
    """ Process the data used in the machine learning pipeline.

    preprocessing the data, handling categorical features via OneHotEncoding, 
    and binarizing the target label.

    Inputs
    ------
    X : pd.DataFrame
        Datafeed.
    categorical_features : list
        List of categorical features.
    label : str
        Name of the label column in X.
    training : bool
        If training is True, new encoder and label binarizer are fitted.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Fitted encoder if training is False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Fitted LabelBinarizer if training is False.

    Returns
    -------
    X : np.array or pd.DataFrame
        Processed data.
    y : np.array
        Binarized labels.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Fitted encoder.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Fitted LabelBinarizer.
    """
    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = None

    X_categorical = X[categorical_features].astype(str)
    X_continuous = X.drop(columns=categorical_features)

    if training:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = pd.DataFrame(encoder.fit_transform(X_categorical))
        if y is not None:
            y = np.atleast_1d(np.squeeze(lb.fit_transform(y.values)))
    else:
        X_categorical = pd.DataFrame(encoder.transform(X_categorical))
        if y is not None:
            y = np.atleast_1d(np.squeeze(lb.transform(y.values)))

    X_categorical.index = X_continuous.index
    X = pd.concat([X_continuous, X_categorical], axis=1)
    
    # Ensure column names are strings
    X.columns = X.columns.astype(str)

    if label is not None:
        return X, y, encoder, lb
    else:
        return X, encoder, lb
