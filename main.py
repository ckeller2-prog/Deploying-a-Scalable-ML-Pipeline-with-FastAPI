import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from ml.data import process_data
from ml.model import inference, load_model

print("Initializing FastAPI app...")

class Data(BaseModel):
    age: int = Field(..., example=39)
    workclass: str = Field(..., example="State-gov")
    fnlwgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., example=13, alias="education-num")
    marital_status: str = Field(..., example="Never-married", alias="marital-status")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=2174, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")

app = FastAPI()

encoder = None
model = None

@app.on_event("startup")
async def startup_event():
    global encoder, model
    print("Loading encoder and model from disk...")
    encoder = load_model("model/encoder.pkl")
    model = load_model("model/model.pkl")
    print("Model and encoder loaded successfully!")

@app.get("/")
async def get_root():
    return {"greeting": "Welcome to the Census Income Prediction API!"}

@app.post("/predict/")
async def post_inference(data: Data):
    data_dict = data.dict(by_alias=True)
    df = pd.DataFrame.from_dict({k: [v] for k, v in data_dict.items()})
    
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

    X_processed, _, _ = process_data(
        df,
        categorical_features=cat_features,
        training=False,
        encoder=encoder
    )
    
    preds = inference(model, X_processed)
    if preds[0] == 1:
        prediction = ">50K"
    else:
        prediction = "<=50K"
        
    return {"result": prediction}