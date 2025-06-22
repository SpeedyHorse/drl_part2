import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def min_max_scale(series, max_value, min_value):
    return (series - min_value) / (max_value - min_value)

type_dict = {
    "Destination Port": np.int64,
    "Protocol": np.int64,
    "Flow Duration": np.int64,
    "Total Fwd Packets": np.int64,
    "Total Backward Packets": np.int64,
    "Total Length of Fwd Packets": np.int64,
    "Total Length of Bwd Packets": np.int64,
    "Fwd Packet Length Max": np.int64,
    "Fwd Packet Length Min": np.int64,
    "Fwd Packet Length Mean": np.float32,
    "Fwd Packet Length Std": np.float32,
    "Bwd Packet Length Max": np.int64,
    "Bwd Packet Length Min": np.int64,
    "Bwd Packet Length Mean": np.float32,
    "Bwd Packet Length Std": np.float32,
    "Flow Bytes/s": np.float32,
    "Flow Packets/s": np.float32,
    "Flow IAT Mean": np.float32,
    "Flow IAT Std": np.float32,
    "Flow IAT Max": np.int64,
    "Flow IAT Min": np.int64,
    "Fwd IAT Total": np.int64,
    "Fwd IAT Mean": np.float32,
    "Fwd IAT Std": np.float32,
    "Fwd IAT Max": np.int64,
    "Fwd IAT Min": np.int64,
    "Bwd IAT Total": np.int64,
    "Bwd IAT Mean": np.float32,
    "Bwd IAT Std": np.float32,
    "Bwd IAT Max": np.int64,
    "Bwd IAT Min": np.int64,
    "Fwd PSH Flags": np.int64,
    "Fwd Header Length": np.int64,
    "Bwd Header Length": np.int64,
    "Fwd Packets/s": np.float32,
    "Bwd Packets/s": np.float32,
    "Min Packet Length": np.int64,
    "Max Packet Length": np.int64,
    "Packet Length Mean": np.float32,
    "Packet Length Std": np.float32,
    "Packet Length Variance": np.float32,
    "SYN Flag Count": np.int64,
    "PSH Flag Count": np.int64,
    "ACK Flag Count": np.int64,
    "Down/Up Ratio": np.float32,
    "Average Packet Size": np.float32,
    "Avg Fwd Segment Size": np.float32,
    "Avg Bwd Segment Size": np.float32,
    "Bwd Avg Packets/Bulk": np.float32,
    "Bwd Avg Bulk Rate": np.int64,
    "Subflow Fwd Packets": np.int64,
    "Subflow Fwd Bytes": np.int64,
    "Subflow Bwd Packets": np.int64,
    "Subflow Bwd Bytes": np.int64,
    "Init_Win_bytes_forward": np.int64,
    "Init_Win_bytes_backward": np.int64,
    "act_data_pkt_fwd": np.int64,
    "min_seg_size_forward": np.int64,
    "Active Mean": np.float32,
    "Active Std": np.float32,
    "Active Max": np.int64,
    "Active Min": np.int64,
    "Idle Mean": np.float32,
    "Idle Std": np.float32,
    "Idle Max": np.int64,
    "Idle Min": np.int64,
}

def get_type(column_label):
    column_type = type_dict[column_label]
    if column_type == np.int64:
        return 0
    elif column_type == np.float32:
        return 1

skip_columns = [
    "Destination Port",
    "Protocol",
    "Label",
    "Attempted Category",
]

origin_df = pd.read_csv("data_cicids2017/1_formated/cicids2017_formated.csv")
generated_df = pd.read_csv("data_cicids2017/3_final/cicids2017_generated_gan.csv")

origin_df = origin_df[origin_df["Attempted Category"] == -1]
label_list = origin_df["Label"].unique().tolist()
generated_df = generated_df[generated_df["Label"].isin(label_list)]

# ----------------------------

# print("start scaling")
# with open("min_max_value.csv", "w") as f:
#     f.write("label,max,min,type,0:int,1:float\n")
#     for column in df.columns:
#         if column in skip_columns:
#             continue
#         else:
#             column_max = df[column].max()
#             column_min = df[column].min()
#         column_type = get_type(column)
#         f.write(f"{column},{column_max},{column_min},{column_type}\n")
#         df[column] = min_max_scale(df[column], column_max, column_min)

concat_list = []
f = open("gan_min_max_value.csv", "w")
f.write("label,max,min,type,0:int,1:float\n")

print("start concat")
for label in label_list:
    print(f"start concat {label}")
    origin_df_label = origin_df[origin_df["Label"] == label]
    generated_df_label = generated_df[generated_df["Label"] == label]
    if len(generated_df_label) == 100_000:
        generated_tmp, _ = train_test_split(generated_df_label, test_size=0.5, random_state=42)
        _, origin_tmp = train_test_split(origin_df_label, test_size=50_000, random_state=42)
        concat_list.append(generated_tmp)
        concat_list.append(origin_tmp)
    else:
        concat_list.append(generated_df_label)
        concat_list.append(origin_df_label)

concat_df = pd.concat(concat_list, ignore_index=True)

print(concat_df["Label"].value_counts())

print("start scaling")
for column in concat_df.columns:
    if column in skip_columns:
        continue
    else:
        column_max = concat_df[column].max()
        column_min = concat_df[column].min()
        column_type = get_type(column)
        f.write(f"{column},{column_max},{column_min},{column_type}\n")

f.close()
print("finish")

concat_df.to_csv("data_cicids2017/3_final/cicids2017_scaled_after_gan.csv", index=False, chunksize=10_000)