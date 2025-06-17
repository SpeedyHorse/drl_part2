import pandas as pd
import numpy as np

df = pd.read_csv("data_cicids2017/1_formated/cicids2017_formated.csv")

columns = df.columns.tolist()

with open("min_max_value.csv", "w") as f:
    f.write("label,min,max,type,0:int,1:float\n")
    for column in columns:
        if column == "Label":
            continue
        if df[column].dtype == np.int64:
            print(f"{column}: {df[column].min()} ~ {df[column].max()}")
        elif df[column].dtype == np.float32:
            print(f"{column}: {df[column].min()} ~ {df[column].max()}")