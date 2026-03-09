import os
import pandas as pd
from fpdf import FPDF

# Helper function to check if text is Latin-1 compatible
def is_latin1_compatible(text):
    try:
        text.encode('latin-1')
        return True
    except UnicodeEncodeError:
        return False

def save_selected_extracted_texts(extracted_texts_csv, df_test_csv, output_main_dir):
    # Load both CSVs
    df_texts = pd.read_csv(extracted_texts_csv)
    df_test = pd.read_csv(df_test_csv)

    # Convert tracking numbers to string for safe matching
    df_texts['Tracking Number'] = df_texts['Tracking Number'].astype(str).str.strip()
    df_texts['Extracted Text'] = df_texts['Extracted Text'].astype(str).fillna('')

    df_test['Tracking Number'] = df_test['Tracking Number'].astype(str).str.strip()

    # Filter: Only keep rows where Tracking Number is present in df_test
    df_texts_filtered = df_texts[df_texts['Tracking Number'].isin(df_test['Tracking Number'])]

    print(f"Found {len(df_texts_filtered)} matching rows to save.")

    if not os.path.exists(output_main_dir):
        os.makedirs(output_main_dir)

    for idx, row in df_texts_filtered.iterrows():
        tracking_number = row['Tracking Number']
        extracted_text = row['Extracted Text'].strip()

        if not extracted_text:
            print(f"Skipping {tracking_number}: Empty extracted text")
            continue

        if not is_latin1_compatible(extracted_text):
            print(f"Skipping {tracking_number}: Unsupported Unicode characters")
            continue

        # Create folder for tracking number
        folder_path = os.path.join(output_main_dir, tracking_number)
        os.makedirs(folder_path, exist_ok=True)

        # Create and save PDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, extracted_text)

        pdf_output_path = os.path.join(folder_path, f"{tracking_number}.pdf")
        pdf.output(pdf_output_path)
        print(f"Saved PDF for Tracking Number: {tracking_number}")

# Example usage:
save_selected_extracted_texts(
    extracted_texts_csv="/Users/vishnucharan/Desktop/Programming/LLM/LangGraph/Project_Corgi/AutoRegressive/GPT_Model/LLM_Data/withDocs/data_exfeatures_docfeatures.csv",
    df_test_csv="/Users/vishnucharan/Desktop/Programming/LLM/LangGraph/Project_Corgi/AutoRegressive/GPT_Model/LLM_Data/withDocs/df_test_withDocs.csv",
    output_main_dir="/Users/vishnucharan/Desktop/Programming/LLM/LangGraph/Project_Corgi/AutoRegressive/GPT_Model/LLM_Data/Clean_Docs"
)
