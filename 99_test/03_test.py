import pandas as pd
from glob import glob

DIR_PATH = "data_cicids2017/0_raw"
files = glob(f"{DIR_PATH}/*.csv")

df = pd.DataFrame()
for file in files:
    df = pd.concat([df, pd.read_csv(file)])

print(df["Label"].value_counts())

print("-"*100)

df = pd.read_csv("data_cicids2017/3_final/cicids2017_formated_scaled.csv")

print(df["Label"].value_counts())