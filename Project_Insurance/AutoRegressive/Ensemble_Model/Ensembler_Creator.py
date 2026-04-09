from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
from constants import SYSTEM_PROMPT, SCALER_FILE
from utils import build_prompt_row_withoutDoc, build_prompt_row_withDoc
import openai
import modal
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from AutoRegressive.DL_Model.predictor_endpoint import app, predict
from tensorflow.keras.models import load_model

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)
GPT_2_jobid = os.getenv("JOB_ID_SFT_MODEL2")
GPT_1_jobid = os.getenv("JOB_ID_SFT_MODEL1")
scaler = joblib.load(SCALER_FILE)

dl_model_path = "/Users/vishnucharan/Desktop/Programming/LLM/LangGraph/Project_Corgi/AutoRegressive/DL_Model/DL_Regression_predictor.h5"
dl_model = load_model(dl_model_path)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore") 


def GPT_promptFormat(claim_data):
    return [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": json.dumps(claim_data)}
    ]

def GPT_Call(claim_data,GPT_jobid):
    try:
        response = client.chat.completions.create(
            model=GPT_jobid,
            messages=GPT_promptFormat(claim_data)
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"\n Error calling LLM API: {e}")
        return None

def make_pred(claim_data,GPT_jobid):
    pred = GPT_Call(claim_data,GPT_jobid)
    if pred is None:
        return None
    try:
        pred_val = float(str(pred).strip().replace("$", "").replace(",", ""))
        return pred_val
    except ValueError:
        print(f"\n Could not parse prediction output: {pred}")
        return None

def GPT_predictor(row,GPT_Type):
    if(GPT_Type=='2'):
        GPT_jobid = GPT_2_jobid
        build_prompt = build_prompt_row_withoutDoc
        if "Extracted Fields" in row.index:
            row = row.drop("Extracted Fields")
    else:
        GPT_jobid = GPT_1_jobid
        build_prompt = build_prompt_row_withDoc

    # trueval = row['Approved Benefit Amount']
    claim_data = SYSTEM_PROMPT + build_prompt(row)
    predval = make_pred(claim_data,GPT_jobid)

    if predval is None:
        print(f"\n Warning: Skipped prediction due to error.")
        return 0.0
    else:
        return predval
    
def DL_Regression_predictor(row):
    row = row.drop("Approved Benefit Amount")
    if "Extracted Fields" in row.index:
        row = row.drop("Extracted Fields")
    row_df = pd.DataFrame([row])
    row_scaled = scaler.transform(row_df)
    row_dict_scaled = dict(zip(row.index, row_scaled.flatten()))
    with modal.enable_output():
        with app.run():
            pred_val = predict.remote(row_dict_scaled)
    return pred_val

def DL_Regression_predictor_local(row):
    row = row.drop("Approved Benefit Amount", errors='ignore')
    if "Extracted Fields" in row.index:
        row = row.drop("Extracted Fields")
    
    row_df = pd.DataFrame([row])
    row_scaled = scaler.transform(row_df)
    pred_val = dl_model.predict(row_scaled)[0]
    return pred_val


#WithDocs Predictor
# GPT_predictor(row,GPT_Type)
# #WithoutDocs Predictor
# GPT_predictor(row,GPT_Type)
# #DL Model Predictor
# DL_Regression_predictor(row)

