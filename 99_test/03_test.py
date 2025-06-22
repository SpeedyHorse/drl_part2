import pandas as pd

origin_df = pd.read_csv("data_cicids2017/1_formated/cicids2017_formated.csv")
generated_df = pd.read_csv("data_cicids2017/3_final/cicids2017_generated_gan.csv")

origin_df = origin_df[origin_df["Attempted Category"] == -1]
label_list = origin_df["Label"].unique().tolist()
generated_df = generated_df[generated_df["Label"].isin(label_list)]

origin_label_count = origin_df["Label"].value_counts()
generated_label_count = generated_df["Label"].value_counts()

for label in origin_label_count.index:
    print(f"{label}: {origin_label_count[label]} + {generated_label_count[label]} = {origin_label_count[label] + generated_label_count[label]}")