# Agentic AI - Property Insurance Benefit Amount Estimator
AI-powered platform for calculating insurance benefits in property management
From a Ycombinator based startup

This project predicts the maximum insurance benefit for property management clients based on historical claims and property-related features. It leverages a multi-model ensemble combining regression and LLM-based models, fine-tuned for structured and unstructured claim data, to provide accurate and reliable benefit estimations.

![Alt text](Project_Corgi/Corgi_data.png)

The system uses property insurance claim records containing both client information and property/lease details including:
| Feature                    | Type        | Encoding / Range |
| -------------------------- | ----------- | ---------------- |
| Monthly rent               | Numeric     | Raw value        |
| Treaty policy frequency    | Categorical | 1–7              |
| Group policy type          | Categorical | 1–7              |
| Third tenant present       | Boolean     | 0/1              |
| Dual tenant                | Boolean     | 0/1              |
| Max benefit as % of rent   | Numeric     | 0–1              |
| Lease duration             | Numeric     | Days             |
| Lease city frequency       | Categorical | 0–80             |
| Days remaining on lease    | Numeric     | Days             |
| Zip code frequency         | Categorical | 0–30             |
| Max benefit amount         | Numeric     | Raw value        |
| Time lived before move-out | Numeric     | Days             |
| Amount of claim            | Numeric     | Raw value        |
| Termination type           | Categorical | 0/1              |


![Alt text](Project_Corgi/Corgi_Agentic.png)

### Fine-tuned **[LLaMA Model on HuggingFace](https://huggingface.co/vishnucharan717/insuranceAmountPredictor-2025-04-25_08.49.57)** for structured features.

Modeling Pipeline

The system leverages a multi-model approach, combining LLM-based models with a traditional regression model to predict insurance benefits. Each model is specialized, and their outputs are combined into an ensemble for robust and accurate predictions.

### Structured LLM Model

- Structured features (numerical and categorical) are converted into string sequences for fine-tuning.

- Fine-tuned on LLaMA and GPT-4o Mini models.

- Quantized LoRA (Low-Rank Adapter) training is used for memory- and compute-efficient parameter updates.

- Training is monitored using Weights & Biases to track loss, convergence, and model performance.

- Captures patterns from structured claim data to generate precise benefit predictions.

### Document + Feature LLM Model

- Combines structured features as strings with claim documents (text) for fine-tuning.

- Enables contextual and semantic understanding beyond raw numerical features.

- Fine-tuned with the same LoRA + quantization approach, tracked via Weights & Biases.

- Supports retrieval-augmented workflows (RAG) when paired with embeddings for richer predictions.

### Regression Model

- Trained on numerical features alone to predict claim amounts.

- Serves as a baseline for ensemble predictions.

- Provides stable, interpretable outputs, complementing the LLM predictions in the ensemble.

### Ensemble Model

- Combines outputs from Structured LLM, Document + Feature LLM, and Regression Model.

- Improves robustness and accuracy by leveraging complementary strengths of LLM reasoning and traditional regression.

- Generates the final predicted insurance benefit with high reliability.

### Training Analysis - Weights and Bias open API - 

![Alt text](Project_Corgi/tr_img1)
![Alt text](Project_Corgi/tr_img2)
![Alt text](Project_Corgi/tr_img3)
