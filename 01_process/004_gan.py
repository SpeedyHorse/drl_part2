import os
import pandas as pd

pd.set_option("display.max_columns", None)

TRAIN_PATH = os.path.abspath("data_cicids2017/1_formated/cicids2017_formated_scaled.csv")
df = pd.read_csv(TRAIN_PATH)
print("load data done")

print(df.head(2))