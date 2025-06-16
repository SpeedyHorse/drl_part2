import pandas as pd
import numpy as np
from glob import glob
from imblearn.pipeline   import Pipeline
from imblearn.under_sampling import NearMiss
from imblearn.combine      import SMOTEENN
from imblearn.over_sampling import SMOTE
import ipaddress as ip
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from util import BASE, CICIDS2017

def majority_sampling(df):
    # undersampling + oversampling
    X = df.drop(columns=['Label'])
    y = df['Label']
    pipeline = Pipeline([
        ('nm', NearMiss(version=1)),
        ('smoteenn', SMOTEENN(
            smote=SMOTE(k_neighbors=2, random_state=42),
            n_jobs=-1))
    ])
    print("majority sampling start")
    X_res, y_res = pipeline.fit_resample(X, y)
    print("majority sampling end")
    resampled = pd.concat([X_res, y_res], axis=1)
    print(resampled["Label"].value_counts())
    return resampled

def minority_sampling(df):
    # oversampling
    X = df.drop(columns=['Label'])
    y = df['Label']
    smote = SMOTE(k_neighbors=1, random_state=42)
    pipeline = Pipeline([
        ('smote', smote)
    ])
    print("minority sampling start")
    X_res, y_res = pipeline.fit_resample(X, y)
    print("minority sampling end")
    resampled = pd.concat([X_res, y_res], axis=1)
    print(resampled["Label"].value_counts())
    return resampled


# --- 1) ファイル一括読み込み＋前処理 ---
def fast_process(df, type="normal"):
    if type == "normal":
        df = df.drop(CICIDS2017().get_delete_columns(), axis=1)
    elif type == "full":
        df = df.drop(['Flow ID','Src IP','Attempted Category'], axis=1)
        # Timestamp→秒
        df['Timestamp'] = (
            pd.to_datetime(df['Timestamp'], format="%Y-%m-%d %H:%M:%S.%f")
            .astype('int64') // 10**9
        )
        # IP文字列→整数
        df['Dst IP'] = df['Dst IP'].apply(lambda x: int(ip.IPv4Address(x)))
        # 欠損／無限大落とし
    return df.replace([np.inf, -np.inf], np.nan).dropna()


print("start")
files = glob("data_cicids2017/0_raw/*.csv")
dfs = [fast_process(pd.read_csv(f)) for f in tqdm(files)]
df  = pd.concat(dfs, ignore_index=True)

rename_dict = {
    k: v for k, v in zip(CICIDS2017().get_features_labels(), BASE().get_features_labels())
}

df = df.rename(columns=rename_dict)

df.to_csv(
    "data_cicids2017/1_formated/cicids2017_formated.csv", 
    index=False,
    chunksize=500_000
)

print("end")
exit()

df, test_df = train_test_split(df, test_size=0.5, random_state=42)

# --- 2) アンダー+オーバーサンプリング ---
label_counts = df["Label"].value_counts()
label = label_counts.iloc[4:5].index[0]
count = label_counts.iloc[4:5].values[0]
print(label, count)

majority_labels = label_counts[label_counts >= count].index
minority_labels = label_counts[label_counts <= count].index

majority_df = df[df["Label"].isin(majority_labels)]
minority_df = df[df["Label"].isin(minority_labels)]

majority_df = majority_sampling(majority_df)
minority_df = minority_sampling(minority_df)

majority_df = majority_df[majority_df["Label"] != label]

df = pd.concat([majority_df, minority_df], axis=0)

print(df["Label"].value_counts())

# --- 3) 一発 CSV 出力 ---
print("saving start")
df.to_csv(
    "data_cicids2017/1_sampling/cicids2017_sampled.csv",
    index=False,
    chunksize=500_000
)
test_df.to_csv(
    "data_cicids2017/1_sampling/cicids2017_sampled_test.csv",
    index=False,
    chunksize=500_000
)
print("saving end")