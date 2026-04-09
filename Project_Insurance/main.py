from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from constants import DOC_PATH_FORTESTING, GENERAL_DATA

from graph.graph import app

if __name__=="__main__":
    df_test = pd.read_csv(GENERAL_DATA)
    row = df_test[df_test['Tracking Number'] == 299]
    row = row.drop(columns=["Extracted Fields"])
    row = row.drop(columns=["Approved Benefit Amount"])
    general_fields = row.iloc[0].to_dict()

    print("Hello Corgi!!")
    print(app.invoke(input={"question":"Get me all relevant property management claim details on - Addendum 364 Elders Pond for predicting Approved Benefit Amount", "doc_path":DOC_PATH_FORTESTING, "general_fields":general_fields}))
