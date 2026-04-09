#This GPT Predictor - Uses only Relevant Fields (Without Extra Docs)

from dotenv import load_dotenv
load_dotenv()

import os
import time
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import openai
from openai import OpenAI
from constants import SYSTEM_PROMPT
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)
GPT_2_jobid = os.getenv("JOB_ID_SFT_MODEL2")

def GPT_2_promptFormat(claim_data):
    return [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": json.dumps(claim_data)}
    ]

def GPT_2_Call(claim_data):
    response = client.chat.completions.create(
        model=GPT_2_jobid,
        messages=GPT_2_promptFormat(claim_data)
    )
    return response.choices[0].message.content

def GPT_2_predictor(claim_data, GPT_2_Call, retries=5):
    """ Returns: tuple: (true_value, predicted_value) or None if failed."""
    last_message = claim_data["messages"].pop()
    true_val = float(last_message["content"].strip())
    for attempt in range(retries):
        try:
            pred = GPT_2_Call(claim_data)
            pred_val = float(str(pred).strip().replace("$", "").replace(",", ""))
            return (true_val, pred_val)
        except openai.RateLimitError:
            wait = 2 ** attempt
            print(f"Rate limit hit. Retrying in {wait}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"Error: {e}")
            return None

    print(f"Failed after {retries}")
    return None

def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

def process_sample(claim_data_sample):
    true_val, pred_val = GPT_2_predictor(claim_data_sample, GPT_2_Call)
    return true_val, pred_val

def safe_predict(sample, retries=5):
    for attempt in range(retries):
        try:
            return process_sample(sample)
        except Exception as e:
            wait = 2 ** attempt
            print(f"⚠️ Error predicting sample. Retrying in {wait}s...")
            time.sleep(wait)
    print("Failed after retries.")
    return None

def main(jsonl_path):
    data = read_jsonl(jsonl_path)

    true_values = []
    predicted_values = []

    num_threads = 4  # Adjust based on your CPU

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(safe_predict, sample) for sample in data]

        for future in tqdm(as_completed(futures), total=len(futures), desc="🔮 Predicting"):
            result = future.result()
            if result:
                true_val, pred_val = result
                true_values.append(true_val)
                predicted_values.append(pred_val)

    true_values = np.array(true_values)
    predicted_values = np.array(predicted_values)

    # Metrics calculation
    mae = mean_absolute_error(true_values, predicted_values)
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_values, predicted_values)

    # MAPE & Accuracy
    nonzero_mask = true_values != 0
    mape = np.mean(np.abs((true_values[nonzero_mask] - predicted_values[nonzero_mask]) / true_values[nonzero_mask])) * 100
    accuracy = 100 - mape

    print("\n🔍 Evaluation Metrics:")
    print(f"MAE   = {mae:.2f}")
    print(f"RMSE  = {rmse:.2f}")
    print(f"R²    = {r2:.4f}")
    print(f"MAPE  = {mape:.2f}%")
    print(f"Approx. Accuracy = {accuracy:.2f}%")

jsonfilepath="/Users/vishnucharan/Downloads/insurance/GPT_Model/WithoutDocs/Test_Data/Insurance_data_test_withoutDocs.jsonl"
main(jsonfilepath)