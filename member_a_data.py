import os
import kagglehub
import pandas as pd
from crewai import LLM
from dotenv import load_dotenv

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")

llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    temperature=0,
    api_key=groq_key
)

print("downloading dataset from kaggle...")
path = kagglehub.dataset_download("ashmitcajla/dataset-for-blood-glucose-level-readings")
filename = os.listdir(path)[0]
df = pd.read_excel(f"{path}/{filename}", engine="openpyxl", skiprows=2)
df.columns = df.columns.str.strip()
df = df.dropna(subset=["Blood Glucose Level(BGL)"])
df = df.reset_index(drop=True)

print(f"dataset loaded!! we have {len(df)} patient records")

def get_blood_glucose(age=None):

    if age is not None:
        matches = df[df["Age"] == age]
        if len(matches) > 0:
            record = matches.sample(1).iloc[0]
        else:
            print(f"no patient found with age {age}, picking random one")
            record = df.sample(1).iloc[0]
    else:
        record = df.sample(1).iloc[0]

    age_val     = record["Age"]
    glucose     = record["Blood Glucose Level(BGL)"]
    sys_bp      = record["Systolic Blood Pressure"]
    dia_bp      = record["Diastolic Blood Pressure"]
    heartrate   = record["Heart Rate"]
    temperature = record["Body Temperature"]
    oxygen      = record["SPO2"]

    if record["Sweating  (Y/N)"] == "Y" or record["Sweating  (Y/N)"] == 1:
        sweating = "Yes"
    else:
        sweating = "No"

    if record["Shivering (Y/N)"] == "Y" or record["Shivering (Y/N)"] == 1:
        shivering = "Yes"
    else:
        shivering = "No"

    if record["Diabetic/NonDiabetic (D/N)"] == "D":
        diabetic = "Diabetic"
    else:
        diabetic = "Non-Diabetic"

    result = f"""
    Patient Info (real data from kaggle dataset):
    ----------------------------------------------
    Age              : {age_val} years
    Blood Glucose    : {glucose} mg/dL
    Blood Pressure   : {sys_bp}/{dia_bp} mmHg
    Heart Rate       : {heartrate} bpm
    Body Temperature : {temperature} F
    Oxygen Level     : {oxygen}%
    Sweating         : {sweating}
    Shivering        : {shivering}
    Diabetic Status  : {diabetic}
    ----------------------------------------------
    """
    return result
