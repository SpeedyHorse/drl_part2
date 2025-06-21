import os
import sys
import pandas as pd


def convert_to_data(df):
    column_data = pd.read_csv("min_max_value.csv")
    for row in column_data.itertuples():
        label = row.label
        max_value = row.max
        min_value = row.min
        type = row.type
        if type == 0:
            df[label] = df[label].apply(lambda x: int(x * (max_value - min_value) + min_value))
        else:
            df[label] = df[label].apply(lambda x: float(x * (max_value - min_value) + min_value))
    return df


def train_gan(df):
    pass


def generate_data():
    pass


if __name__ == "__main__":
    # args = sys.argv
    # if len(args) != 2:
    #     print("Usage: python 004_gan.py <is_train> <is_generate>")
    #     sys.exit(1)
    # else:
    #     print(f"is_train: {args[1]}, is_generate: {args[2]}")
    #     is_train = args[1] == "y"
    #     is_generate = args[2] == "y"

    pd.set_option("display.max_columns", None)

    TRAIN_PATH = os.path.abspath("data_cicids2017/3_final/cicids2017_formated_scaled.csv")
    df = pd.read_csv(TRAIN_PATH)
    print("load data done")

    print(df.head(2))
    print("-" * 100)
    df = convert_to_data(df)
    print(df.head(2))

    # if is_train:
    #     print("start train")
    #     train_gan(df)
    
    # if is_generate:
    #     print("start generate")
    #     generate_data()