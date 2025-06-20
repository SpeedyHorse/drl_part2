import os
import pandas as pd
import numpy as np

TRAIN_PATH = os.path.abspath("data_cicids2017/1_formated/cicids2017_formated.csv")
df = pd.read_csv(TRAIN_PATH)

def min_max_scale(series, max_value, min_value):
    return (series - min_value) / (max_value - min_value)

type_dict = {
    "Destination Port": np.int64,
    "Protocol": np.int64,
    "Flow Duration": np.float32,
    "Total Fwd Packets": np.float32,
    "Total Backward Packets": np.float32,
    "Total Length of Fwd Packets": np.float32,
    "Total Length of Bwd Packets": np.float32,
    "Fwd Packet Length Max": np.float32,
    "Fwd Packet Length Min": np.float32,
    "Fwd Packet Length Mean": np.float32,
    "Fwd Packet Length Std": np.float32,
    "Bwd Packet Length Max": np.float32,
    "Bwd Packet Length Min": np.float32,
    "Bwd Packet Length Mean": np.float32,
    "Bwd Packet Length Std": np.float32,
    "Flow Bytes/s": np.float32,
    "Flow Packets/s": np.float32,
    "Flow IAT Mean": np.float32,
    "Flow IAT Std": np.float32,
    "Flow IAT Max": np.float32,
    "Flow IAT Min": np.float32,
    "Fwd IAT Total": np.float32,
    "Fwd IAT Mean": np.float32,
    "Fwd IAT Std": np.float32,
    "Fwd IAT Max": np.float32,
    "Fwd IAT Min": np.float32,
    "Bwd IAT Total": np.float32,
    "Bwd IAT Mean": np.float32,
    "Bwd IAT Std": np.float32,
    "Bwd IAT Max": np.float32,
    "Bwd IAT Min": np.float32,
    "Fwd PSH Flags": np.float32,
    "Fwd Header Length": np.float32,
    "Bwd Header Length": np.float32,
    "Fwd Packets/s": np.float32,
    "Bwd Packets/s": np.float32,
    "Min Packet Length": np.float32,
    "Max Packet Length": np.float32,
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
    "Bwd Avg Bulk Rate": np.float32,
    "Subflow Fwd Packets": np.float32,
    "Subflow Fwd Bytes": np.float32,
    "Subflow Bwd Packets": np.float32,
    "Subflow Bwd Bytes": np.float32,
    "Init_Win_bytes_forward": np.float32,
    "Init_Win_bytes_backward": np.float32,
    "act_data_pkt_fwd": np.float32,
    "min_seg_size_forward": np.float32,
    "Active Mean": np.float32,
    "Active Std": np.float32,
    "Active Max": np.float32,
    "Active Min": np.float32,
    "Idle Mean": np.float32,
    "Idle Std": np.float32,
    "Idle Max": np.float32,
    "Idle Min": np.float32,
}

def get_type(column_label):
    column_type = type_dict[column_label]
    if column_type == np.int64:
        return 0
    elif column_type == np.float32:
        return 1

skip_columns = ["Destination Port", "Protocol", "Label", "Attempted Category"]

with open("min_max_value.csv", "w") as f:
    f.write("label,max,min,type,0:int,1:float\n")
    for column in df.columns:
        if column in skip_columns:
            continue
        else:
            column_max = df[column].max()
            column_min = df[column].min()
        column_type = get_type(column)
        f.write(f"{column},{column_max},{column_min},{column_type}\n")
        df[column] = min_max_scale(df[column], column_max, column_min)

df.to_csv("data_cicids2017/3_final/cicids2017_formated_scaled.csv", index=False)