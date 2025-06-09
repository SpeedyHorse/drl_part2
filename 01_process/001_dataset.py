import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from collections import Counter
from glob import glob
from tqdm import tqdm
import numpy as np

def process_data(df, is_drop=True):
    df = df.drop(columns=[
        "Flow ID",
        "Src IP",
    ])

    if is_drop:
        df = df.drop(columns=[
            "Dst IP",
            "Timestamp"
        ])
    else:
        # Timestamp
        import datetime
        # 2017-07-07 11:59:50.315195 -> 1500000000.315195
        DATEFORMAT = "%Y-%m-%d %H:%M:%S.%f"
        df["continuous_timestamp"] = df["Timestamp"].apply(
            lambda x: datetime.strptime(x, DATEFORMAT).timestamp()
        )
        df = df.drop(columns=[
            "Timestamp",
        ])
        df = df.rename(columns={
            "continuous_timestamp": "Timestamp"
        })
        # Dst IP
        import ipaddress as ip
        df["destination_ip"] = df["Dst IP"].apply(
            lambda x: int(ip.IPv4Address(x))
        )
        df = df.drop(columns=[
            "Dst IP",
        ])
        df = df.rename(columns={
            "destination_ip": "Dst IP"
        })
    return df
    
directory_path = "data_cicids2017/0_raw"

df = pd.DataFrame()
files_path = glob(f"{directory_path}/*.csv")
for file_path in tqdm(files_path):
    df_tmp = pd.read_csv(file_path)
    df_tmp = process_data(df_tmp)
    df_tmp = df_tmp.replace([np.inf, -np.inf], np.nan)
    df_tmp = df_tmp.dropna()
    df = pd.concat([df, df_tmp], axis=0)

X = df.drop(columns=["Label", "Attempted Category"])
y = df["Label"]

smote_enn = SMOTEENN(
    random_state=42,
    n_jobs=-1,
    smote=SMOTE(
        k_neighbors=2,
        random_state=42,
    )
)
X_res, y_res = smote_enn.fit_resample(X, y)

print("After: ", Counter(y_res))

df = pd.concat([X_res, y_res], axis=1)

length = len(df)

ROW_COUNTER = 500_000

i = 0
counter = 0
while i < length:
    counter += 1
    if i + ROW_COUNTER > length:
        df_temp = df.iloc[i:length]
        print(f"{i:10d} - {length:10d}")
    else:
        df_temp = df.iloc[i:i + ROW_COUNTER]
        print(f"{i:10d} - {i + ROW_COUNTER:10d}")

    df_temp.to_csv(f"data_cicids2017/1_sampling/{counter:03d}_cicids2017.csv", index=False)
    i += ROW_COUNTER