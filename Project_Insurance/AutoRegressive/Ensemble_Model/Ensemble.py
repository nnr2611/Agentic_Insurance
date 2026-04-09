from Ensembler_Creator import GPT_predictor, DL_Regression_predictor
from utils import build_prompt_row_withoutDoc, build_prompt_row_withDoc
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import os
from constants import TEST_FILE, VAL_FILE
import joblib

GPT_2_jobid = os.getenv("JOB_ID_SFT_MODEL2")
GPT_1_jobid = os.getenv("JOB_ID_SFT_MODEL1")

df_test = pd.read_csv(VAL_FILE)
LLM_withDocs_pred = []
LLM_withoutDocs_pred = []
DL_Regression_pred = []
True_ApprAmount = []

for i in range(len(df_test)):
    row = df_test.iloc[i]
    ApprAmount = row["Approved Benefit Amount"]
    True_ApprAmount.append(ApprAmount)
    LLM_withDocs_pred.append(GPT_predictor(row,"1"))
    LLM_withoutDocs_pred.append(GPT_predictor(row,"2"))
    DL_Regression_pred.append(DL_Regression_predictor(row))

Min_error = []
Max_error = []

for i in range(len(True_ApprAmount)):
    true_val = True_ApprAmount[i]
    preds = [LLM_withDocs_pred[i], LLM_withoutDocs_pred[i], DL_Regression_pred[i]]
    abs_errors = [abs(pred - true_val) for pred in preds]
    Min_error.append(min(abs_errors))
    Max_error.append(max(abs_errors))

X = pd.DataFrame({
    'LLM_withDocs': LLM_withDocs_pred,
    'LLM_withoutDocs': LLM_withoutDocs_pred,
    'DL_Regression': DL_Regression_pred,
    'Min_error': Min_error,
    'Max_error': Max_error,
})

y = pd.Series(True_ApprAmount)

np.random.seed(42)
lr = LinearRegression()
lr.fit(X, y)

feature_columns = X.columns.tolist()

for feature, coef in zip(feature_columns, lr.coef_):
    print(f"{feature}: {coef:.2f}")
print(f"Intercept = {lr.intercept_:.2f}")

joblib.dump(lr, "ensemble_model.h5")
joblib.dump(lr, "ensemble_model.pkl")

