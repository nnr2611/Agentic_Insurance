import modal
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tqdm import tqdm
import joblib
from predictor_endpoint import app, predict

df_test = pd.read_csv("/Users/vishnucharan/Downloads/insurance/GPT_Model/WithoutDocs/CSV/df_test_withoutDocs.csv")

X_test = df_test.drop(columns=["Approved Benefit Amount"])
y_test = df_test["Approved Benefit Amount"].values

y_pred = []

scaler = joblib.load("/Users/vishnucharan/Desktop/Programming/LLM/LangGraph/Project_Corgi/DL_Model/scaler.pkl")

with modal.enable_output():
    with app.run():
        for idx, row in tqdm(X_test.iterrows(), total=len(X_test), desc="Predicting"):
            row_df = pd.DataFrame([row])
            row_scaled = scaler.transform(row_df)
            row_dict_scaled = dict(zip(X_test.columns, row_scaled.flatten()))
            prediction = predict.remote(row_dict_scaled)
            y_pred.append(prediction)

y_pred = np.array(y_pred)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

nonzero_mask = y_test != 0
mape = np.mean(np.abs((y_test[nonzero_mask] - y_pred[nonzero_mask]) / y_test[nonzero_mask])) * 100
accuracy = 100 - mape

print("\n🔍 Evaluation Metrics:")
print(f"MAE   = {mae:.2f}")
print(f"RMSE  = {rmse:.2f}")
print(f"R²    = {r2:.4f}")
print(f"MAPE  = {mape:.2f}%")
print(f"Approx. Accuracy = {accuracy:.2f}%")
