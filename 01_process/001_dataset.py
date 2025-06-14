import pandas as pd
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from collections import Counter
from glob import glob
from tqdm import tqdm
import numpy as np
import datetime

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

def sampling(df):
    # undersampling ... more than thrid
    # 各ラベルの出現回数を計算
    label_counts = df["Label"].value_counts()
    print(label_counts)

    # 100,000より多いラベルと少ないラベルに分類
    more_than_100k_labels = label_counts[label_counts > 100_000].index
    less_than_100k_labels = label_counts[label_counts <= 100_000].index

    # データを分類
    under_df = df[df["Label"].isin(more_than_100k_labels)]
    other_df = df[df["Label"].isin(less_than_100k_labels)]

    cnn = CondensedNearestNeighbour(
        random_state=42,
        n_neighbors=2,
        n_jobs=-1
    )
    print("undersampling start...")
    under_x, under_y = cnn.fit_resample(under_df.drop(columns=["Label"]), under_df["Label"])
    under_res = pd.concat([under_x, under_y], axis=1)

    df = pd.concat([under_res, other_df], axis=0)
    print("undersampling end...")

    # oversampling ... less than third
    smote_enn = SMOTEENN(
        random_state=42,
        n_jobs=-1,
        smote=SMOTE(
            k_neighbors=2,
            random_state=42,
        )
    )
    print("oversampling start...")
    over_x, over_y = smote_enn.fit_resample(df.drop(columns=["Label"]), df["Label"])
    over_res = pd.concat([over_x, over_y], axis=1)
    print("oversampling end...")

    with open("sampling_result.txt", "w") as f:
        f.write(f"sampling: {datetime.datetime.now()}\n")
        f.write("="*100 + "\n")
        f.write(f"undersampling\n")
        for label, count in under_res["Label"].value_counts().items():
            f.write(f"{label}: {count}\n")
        f.write("="*100 + "\n")
        f.write(f"oversampling\n")
        for label, count in over_res["Label"].value_counts().items():
            f.write(f"{label}: {count}\n")
        f.write("="*100 + "\n")

    return over_res


directory_path = "data_cicids2017/0_raw"

df = pd.DataFrame()
files_path = glob(f"{directory_path}/*.csv")

for file_path in tqdm(files_path):
    df_tmp = pd.read_csv(file_path)
    df_tmp = process_data(df_tmp)
    df_tmp = df_tmp.replace([np.inf, -np.inf], np.nan)
    df_tmp = df_tmp.dropna()
    df = pd.concat([df, df_tmp], axis=0)
    # break

df = df.drop(columns=["Attempted Category"])

df = sampling(df)

print("After: ", Counter(df["Label"]))

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

    # df_temp.to_csv(f"data_cicids2017/1_sampling/{counter:03d}_cicids2017.csv", index=False)
    i += ROW_COUNTER