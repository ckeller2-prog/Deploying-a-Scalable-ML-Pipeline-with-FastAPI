import os 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from ml.data import process_data 
from ml.model import train_model, inference, compute_model_metrics, save_model, performance_on_categorical_slice 
#  Load the census.csv data 
data_path = "data/census.csv" 
if not os.path.exists(data_path): 
    # Fallback path check 
    data_path = "Project-Build-an-ML-Pipeline-Starter/data/census.csv" 

column_names = [ 
    "age", 
    "workclass", 
    "fnlwgt", 
    "education", 
    "education-num", 
    "marital-status", 
    "occupation", 
    "relationship", 
    "race", 
    "sex", 
    "capital-gain", 
    "capital-loss", 
    "hours-per-week", 
    "native-country", 
    "income", 
]

data = pd.read_csv(data_path, header=None, names=column_names, skipinitialspace=True)
# Split data into training and test datasets 
train, test = train_test_split(data, test_size=0.20, random_state=42, stratify=data['income']) 
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

# Process the training and test data 
X_train, y_train, encoder, lb = process_data( 
    train, categorical_features=cat_features, label="income", training=True 
) 
X_test, y_test, _, _ = process_data( 
    test, categorical_features=cat_features, label="income", training=False, encoder=encoder, lb=lb 
) 

# Train the model 
print("Training model...") 
model = train_model(X_train, y_train, random_state=42) 
# Run inference on test data 
preds = inference(model, X_test) 
precision, recall, fbeta = compute_model_metrics(y_test, preds) 
print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {fbeta:.4f}") 

# Save model and encoder 
os.makedirs("model", exist_ok=True) 
save_model(model, "model/model.pkl") 
save_model(encoder, "model/encoder.pkl") 
print("Model saved to model/model.pkl") 
print("Encoder saved to model/encoder.pkl") 

# Compute performance on categorical slices and save to slice_output.txt 
slice_outputs = [] 
for feature in cat_features: 
    slice_metrics = performance_on_categorical_slice(test, feature, "income", model, encoder, lb) 
    for val, metrics in slice_metrics.items(): 
        line = f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['fbeta']}\n{feature}: {val}, Count: {metrics['count']}\n" 
        slice_outputs.append(line) 
with open("slice_output.txt", "w") as f: 
    f.writelines(slice_outputs) 
print("Slice performance metrics saved to slice_output.txt")
