import os
import sys
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import time


Z_DIM = 100
NUM_EPOCHS = 10_000
HIDDEN_SIZE = 100
LEARNING_RATE = 0.0002
BATCH_SIZE = 16


class DataFrameDataset(Dataset):
    def __init__(self, df, feature_cols, preprocess_fn=None):
        self.feature_cols = feature_cols
        self.preprocess_fn = preprocess_fn

        if preprocess_fn:
            self.data = preprocess_fn(df[feature_cols])
        else:
            self.data = df[feature_cols].values
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)


class Generator(nn.Module):
    def __init__(self, z_dim, data_column_size, gen_hidden_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_dim, gen_hidden_size)
        self.fc2 = nn.Linear(gen_hidden_size, gen_hidden_size)
        self.fc3 = nn.Linear(gen_hidden_size, data_column_size)

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = F.tanh(self.fc2(x))
        return F.sigmoid(self.fc3(x))


class Discriminator(nn.Module):
    def __init__(self, data_column_size, disc_hidden_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(data_column_size, disc_hidden_size)
        self.fc2 = nn.Linear(disc_hidden_size, disc_hidden_size)
        self.fc3 = nn.Linear(disc_hidden_size, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(self.fc2(x))
        return F.sigmoid(self.fc3(x))


def plot_losses(d_losses, g_losses, title):
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label="Discriminator Loss")
    plt.plot(g_losses, label="Generator Loss")
    plt.title(title)
    plt.legend()
    plt.savefig(f"losses_{title}.png")


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


def train_step(gen_model, disc_model, real_data, device):
    z = torch.randn(BATCH_SIZE, Z_DIM, device=device)

    fake_data = gen_model(z)
    fake_output = disc_model(fake_data)
    real_output = disc_model(real_data)

    d_loss_fake = F.binary_cross_entropy(
        fake_output, 
        torch.zeros_like(fake_output, device=device),
    )
    d_loss_real = F.binary_cross_entropy(
        real_output, 
        torch.ones_like(real_output, device=device),
    )
    d_loss = d_loss_real + d_loss_fake

    g_loss = F.binary_cross_entropy(
        fake_output, 
        torch.ones_like(fake_output, device=device),
    )

    return d_loss, g_loss


def train_gan(df):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    columns = df.columns.tolist()
    feature_cols = [col for col in columns if col != "Label" and col != "Attempted Category"]

    label_list = df["Label"].unique().tolist()
    if len(label_list) != 1:
        raise ValueError(f"Label must be unique: {label_list}")

    dataset = DataFrameDataset(
        df,
        feature_cols,
        preprocess_fn=None
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )

    data_column_size = len(feature_cols)
    gen_hidden_size = disc_hidden_size = HIDDEN_SIZE
    gen_lr = disc_lr = LEARNING_RATE

    gen_model = Generator(Z_DIM, data_column_size, gen_hidden_size).to(device)
    disc_model = Discriminator(data_column_size, disc_hidden_size).to(device)

    # PyTorch 2.0+ であれば torch.compile でモデルを高速化
    if hasattr(torch, "compile"):
        gen_model = torch.compile(gen_model)
        disc_model = torch.compile(disc_model)

    gen_optimizer = optim.Adam(gen_model.parameters(), lr=gen_lr)
    disc_optimizer = optim.Adam(
        disc_model.parameters(), 
        lr=disc_lr,
        betas=(0.5, 0.999),
    )

    d_losses = []
    g_losses = []

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        for i, real_data in enumerate(dataloader):
            real_data = real_data.to(device)
            current_batch_size = real_data.size(0)

            # ---------------------
            #  Discriminatorの学習
            # ---------------------
            disc_optimizer.zero_grad()
            
            z = torch.randn(current_batch_size, Z_DIM, device=device)
            fake_data = gen_model(z)

            # 本物データに対する損失
            real_output = disc_model(real_data)
            d_loss_real = F.binary_cross_entropy(real_output, torch.ones_like(real_output))

            # 偽データに対する損失 (Generatorの勾配は計算しない)
            fake_output = disc_model(fake_data.detach())
            d_loss_fake = F.binary_cross_entropy(fake_output, torch.zeros_like(fake_output))

            # Discriminatorの損失
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            disc_optimizer.step()

            # -----------------
            #  Generatorの学習
            # -----------------
            gen_optimizer.zero_grad()
            
            # Discriminatorの学習で使った偽データを再利用し、Generatorの勾配を計算
            fake_output_for_g = disc_model(fake_data)
            g_loss = F.binary_cross_entropy(fake_output_for_g, torch.ones_like(fake_output_for_g))
            g_loss.backward()
            gen_optimizer.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

            if (i + 1) % 1_000 == 0:
                time_elapsed = time.time() - start_time
                print(f"Step {i + 1:10d} / {len(dataloader):10d} | Time elapsed: {time_elapsed:.2f}s")
                # print(f"\rEpoch {epoch + 1:10d} / {NUM_EPOCHS:10d} | Step {i + 1:10d} / {len(dataloader):10d} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}", end="")

        end_time = time.time()
        print(f"\rEpoch {epoch + 1:10d} / {NUM_EPOCHS:10d} | Time: {end_time - start_time:.2f}s")

        if (epoch + 1) % 10_000 == 0:
            title = f"Epoch {epoch + 1:10d} / {NUM_EPOCHS:10d}"
            plot_losses(d_losses, g_losses, title)
            print(f"\rEpoch {epoch + 1:10d} / {NUM_EPOCHS:10d} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
    
    label = label_list[0]
    torch.save(gen_model.state_dict(), f"result/gan/gen/gen_model_{label}.pth")
    torch.save(disc_model.state_dict(), f"result/gan/disc/disc_model_{label}.pth")


def generate_data(df, max_size=10_000):
    columns = df.columns.tolist()
    feature_cols = [col for col in columns if col != "Label" and col != "Attempted Category"]

    label_list = df["Label"].unique().tolist()
    if len(label_list) != 1:
        raise ValueError(f"Label must be unique: {label_list}")
    label = label_list[0]

    data_column_size = len(feature_cols)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    gen_model = Generator(Z_DIM, data_column_size, HIDDEN_SIZE)
    gen_model.load_state_dict(torch.load(f"result/gan/gen/gen_model_{label}.pth"))
    gen_model.to(device)
    
    if len(df) < max_size:
        z = torch.randn(max_size - len(df), Z_DIM, device=device)
        fake_data = gen_model(z)
    else:
        z = torch.randn(max_size, Z_DIM, device=device)
        fake_data = gen_model(z)

    df = pd.DataFrame(fake_data.detach().cpu().numpy(), columns=feature_cols)
    df["Label"] = label
    df["Attempted Category"] = -2
    return df


if __name__ == "__main__":
    args = sys.argv
    if len(args) != 3:
        print("Usage: python 004_gan.py <is_train> <is_generate>")
        sys.exit(1)
    else:
        is_train = args[1] == "y"
        is_generate = args[2] == "y"
        print(f"is_train: {is_train}, is_generate: {is_generate}")

    pd.set_option("display.max_columns", None)

    TRAIN_PATH = os.path.abspath("data_cicids2017/3_final/cicids2017_formated_scaled.csv")
    df = pd.read_csv(TRAIN_PATH)
    print("load data done")

    if is_train:
        label_list = df["Label"].unique().tolist()
        for label in label_list:
            print(f"start train {label}")
            df_label = df[df["Label"] == label]
            train_gan(df_label)
            print(f"finish train {label}")
    
    if is_generate:
        print("start generate")
        df = generate_data(df, max_size=100_000)
        print(df.head(2))
        print("-" * 100)
        df.to_csv(
            "data_cicids2017/3_final/cicids2017_generated_gan.csv",
            index=False,
            chunksize=10_000,
        )