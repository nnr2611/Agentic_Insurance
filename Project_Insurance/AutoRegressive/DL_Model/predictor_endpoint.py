# predict_endpoint.py

import modal
import pandas as pd
from tensorflow.keras.models import load_model

app = modal.App("DL-Regression-Predictor")
image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "tensorflow", "pandas"
).add_local_dir(
    local_path="/Users/vishnucharan/Desktop/Programming/LLM/llm_engineering/week8/Testing/model",
    remote_path="/Predictor_model"
)

MODEL_PATH = "/Predictor_model/DL_Regression_predictor.h5"

@app.function(image=image)
def predict(row_json: dict) -> float:
    if not hasattr(predict, "model"):
        predict.model = load_model(MODEL_PATH)

    df = pd.DataFrame([row_json])
    prediction = predict.model.predict(df)
    return float(prediction[0][0])
