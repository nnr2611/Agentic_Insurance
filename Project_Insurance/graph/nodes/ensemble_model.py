from typing import Any, Dict
from graph.state import ClaimGraphState
import pandas as pd
import joblib
from AutoRegressive.Ensemble_Model.Ensembler_Creator import GPT_predictor, DL_Regression_predictor, DL_Regression_predictor_local

def ensemble_model(state: ClaimGraphState) -> Dict[str, Any]:
    print("-- ENSEMBLE PREDICT --")
    
    general_fields = state['general_fields']
    row = pd.DataFrame([general_fields]).iloc[0]
    LLM_pred_with_doc = GPT_predictor(row, "1")
    LLM_pred_without_doc = GPT_predictor(row, "2")
    pred_dl = DL_Regression_predictor_local(row)

    preds = [LLM_pred_with_doc, LLM_pred_without_doc, pred_dl]
    min_error = min(abs(p - LLM_pred_with_doc) for p in preds)
    max_error = max(abs(p - LLM_pred_with_doc) for p in preds)

    ensemble_feature_input = pd.DataFrame([{
        "LLM_withDocs": LLM_pred_with_doc,
        "LLM_withoutDocs": LLM_pred_without_doc,
        "DL_Regression": pred_dl,
        "Min_error": min_error,
        "Max_error": max_error,
    }])

    ensemble_model_path = "/Users/vishnucharan/Desktop/Programming/LLM/LangGraph/Project_Corgi/AutoRegressive/Ensemble_Model/new/ensemble_model.pkl"
    ensemble_model = joblib.load(ensemble_model_path)

    final_prediction = ensemble_model.predict(ensemble_feature_input)[0]

    # Update state with final prediction
    return {
        "prediction": final_prediction,
        "general_fields": general_fields,
        "question": state["question"],
        "original_docs": state.get("original_docs", ""),
        "chunk_docs": state["chunk_docs"],
        "doc_path": state.get("doc_path", ""),
        "has_extra_docs": state.get("has_extra_docs", True),
    }
