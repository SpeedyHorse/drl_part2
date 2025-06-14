from flow_package.multi_flow_env import MultipleFlowEnv, InputType
import flow_package as f_p
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
from glob import glob
import dask.dataframe as dd
import gc
import seaborn as sns
import matplotlib
import random
from collections import deque, namedtuple
from itertools import count
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
import torch.optim as optim
from tqdm import tqdm


CONST = f_p.Const()

TRAIN_DIR = os.path.abspath("data_cicids2017/1_sampling")
paths = glob(os.path.join(TRAIN_DIR, "cicids2017_sampled.csv"))

df = pd.DataFrame()
for path in tqdm(paths):

    df = pd.concat([df, pd.read_csv(path, dtype=CONST.dtypes)])

df = df.dropna(how="any").dropna(axis=1, how="any")

del df
gc.collect()

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_input = InputType(
    data=train_df,
    sample_size=5000,
    normalize_exclude_columns=["Protocol", "Destination Port"],
    exclude_columns=["Attempted Category"]
)
train_env = MultipleFlowEnv(train_input)

test_input = InputType(
    data=test_df,
    sample_size=5000,
    is_test=True,
    normalize_exclude_columns=["Protocol", "Destination Port"],
    exclude_columns=["Attempted Category"]
)
test_env = MultipleFlowEnv(test_input)

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

device_name = "cpu"
if True:
    if torch.cuda.is_available():
        device_name = "cuda"
    elif torch.mps.is_available():
        device_name = "mps"
    elif torch.mtia.is_available():
        device_name = "mtia"
    elif torch.xpu.is_available():
        device_name = "xpu"

device = torch.device(device_name)
print(f"device: {device_name}")

Transaction = namedtuple('Transaction', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transaction(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def moving_average(data, window_size):
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')


def plot_graph(data: list, show_result=False):
    plt.figure(figsize=(15, 5))
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")

    means = moving_average(data, 200)
    lines = np.full(len(means), 1 / 15)

    first = len(data) - len(means)
    if first < 0:
        first = 0
    else:
        means = np.concatenate((np.full(first, np.nan), means))
        lines = np.concatenate((np.full(first, np.nan), lines))

    plt.plot(lines, color="blue", linestyle="--")
    plt.xlabel("Episode")
    plt.ylabel("ratio")
    plt.plot(means, color="red")
    plt.grid()
    plt.pause(0.001)
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def plot_normal_graph(data: list, show_result=False):
    plt.figure(figsize=(15, 5))
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")

    means = moving_average(data, 200)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.plot(means, color="red")
    plt.grid()
    plt.pause(0.001)
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def plot_double_graph(data1: list, data2: list, show_result=False):
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    if show_result:
        ax1.set_title("Result")
    else:
        ax1.cla()
        ax1.set_title("Training...")
    means_ax1 = moving_average(data1, 200)
    lines_ax1 = np.full(len(means_ax1), 1 / 15)
    first_ax1 = len(data1) - len(means_ax1)
    if first_ax1 < 0:
        first_ax1 = 0
    else:
        means_ax1 = np.concatenate((np.full(first_ax1, np.nan), means_ax1))
        lines_ax1 = np.concatenate((np.full(first_ax1, np.nan), lines_ax1))
    ax1.plot(lines_ax1, color="blue", linestyle="--")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("ratio")
    ax1.plot(means_ax1, color="red")
    ax2 = fig.add_subplot(1, 2, 2)
    if show_result:
        ax2.set_title("Result")
    else:
        ax2.cla()
        ax2.set_title("Training...")
    means_ax2 = moving_average(data2, 200)
    first_ax2 = len(data2) - len(means_ax2)
    if first_ax2 < 0:
        first_ax2 = 0
    else:
        means_ax2 = np.concatenate((np.full(first_ax2, np.nan), means_ax2))
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Loss")
    ax2.plot(means_ax2, color="red")
    fig.tight_layout()
    plt.pause(0.001)
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def plot_metrics(metrics_dict: dict, show_result=False):
    fig = plt.figure(figsize=(16, 20))
    ac = fig.add_subplot(5, 1, 1)
    ac.plot(metrics_dict["accuracy"], label="accuracy")
    ac.grid()
    ac.set_title("Accuracy")
    pr = fig.add_subplot(5, 1, 2)
    pr.plot(metrics_dict["precision"], label="precision", color="green")
    pr.grid()
    pr.set_title("Precision")
    re = fig.add_subplot(5, 1, 3)
    re.plot(metrics_dict["recall"], label="recall", color="red")
    re.grid()
    re.set_title("Recall")
    f1 = fig.add_subplot(5, 1, 4)
    f1.plot(metrics_dict["f1"], label="f1", color="black")
    f1.grid()
    f1.set_title("F1")
    fpr = fig.add_subplot(5, 1, 5)
    fpr.plot(metrics_dict["fpr"], label="fpr", color="purple")
    fpr.grid()
    fpr.set_title("FPR")
    plt.tight_layout()
    plt.pause(0.001)
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def calculate_metrics(tp, tn, fp, fn):
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp) if tp + fp != 0 else -1
    recall = tp / (tp + fn) if tp + fn != 0 else -1
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else None
    fpr = fp / (fp + tn) if fp + tn != 0 else None
    if precision < 0:
        precision = None
    if recall < 0:
        recall = None
    return accuracy, precision, recall, f1, fpr


PORT_DIM = 32


class DeepFlowNetwork(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(DeepFlowNetwork, self).__init__()
        self.protocol_embedding = nn.Embedding(256, 8)
        self.port_embedding = nn.Embedding(65536, PORT_DIM)
        n_inputs = n_inputs + 6 + PORT_DIM
        self.fc1 = nn.Linear(n_inputs, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, n_outputs)

    def forward(self, x):
        port_emb = self.port_embedding(x[0].long())
        protocol_emb = self.protocol_embedding(x[1].long())
        renew = torch.cat([port_emb, protocol_emb, x[2]], dim=1)
        renew = F.relu(self.fc1(renew))
        renew = F.relu(self.fc2(renew))
        return self.fc3(renew)


UPDATE_TARGET_STEPS = 200
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 100000
TAU = 0.005
LR = 1e-4
REWARD_MATRIX = np.array([
    [1., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2.],
    [-2., 1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
    [-2., -1., 1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
    [-2., -1., -1., 1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
    [-2., -1., -1., -1., 1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
    [-2., -1., -1., -1., -1., 1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
    [-2., -1., -1., -1., -1., -1., 1., -1., -1., -1., -1., -1., -1., -1., -1.],
    [-2., -1., -1., -1., -1., -1., -1., 1., -1., -1., -1., -1., -1., -1., -1.],
    [-2., -1., -1., -1., -1., -1., -1., -1., 1., -1., -1., -1., -1., -1., -1.],
    [-2., -1., -1., -1., -1., -1., -1., -1., -1., 1., -1., -1., -1., -1., -1.],
    [-2., -1., -1., -1., -1., -1., -1., -1., -1., -1., 1., -1., -1., -1., -1.],
    [-2., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., 1., -1., -1., -1.],
    [-2., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., 1., -1., -1.],
    [-2., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., 1., -1.],
    [-2., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., 1.]
])

def get_reward(action, answer):
    return REWARD_MATRIX[action, answer]

num_episodes = 100
n_actions = train_env.action_space.n
n_inputs = train_env.observation_space.shape[0]

state = train_env.reset()

policy_net = DeepFlowNetwork(n_inputs, n_actions).to(device)
target_net = DeepFlowNetwork(n_inputs, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
steps_done = 0

memory = ReplayMemory(10000)
episode_rewards = []
episode_precision = []
loss_array = []

with open("train.log", "w") as f:
    f.write("episode, show\n")

def select_action(state_tensor: torch.Tensor):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state_tensor).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], dtype=torch.long, device=device)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transaction(*zip(*transitions))
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool
    )
    state_batch_port = torch.cat([s[0] for s in batch.state])
    state_batch_protocol = torch.cat([s[1] for s in batch.state])
    state_batch_other = torch.cat([s[2] for s in batch.state])
    state_batch = [state_batch_port, state_batch_protocol, state_batch_other]
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    non_final_next_states_port = torch.cat([s[0] for s in batch.next_state if s is not None])
    non_final_next_states_protocol = torch.cat([s[1] for s in batch.next_state if s is not None])
    non_final_next_states_features = torch.cat([s[2] for s in batch.next_state if s is not None])
    non_final_next_states = [non_final_next_states_port, non_final_next_states_protocol, non_final_next_states_features]
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    expected_state_action_values = reward_batch + GAMMA * next_state_values
    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    utils.clip_grad_value_(policy_net.parameters(), 1000)
    optimizer.step()
    return loss.item()

for i_episode in range(num_episodes):
    test = []
    random.seed(i_episode)
    sum_reward = 0
    confusion_matrix = np.zeros((2, 2), dtype=int)
    initial_state = train_env.reset()
    state = f_p.to_tensor(initial_state, device=device)
    show = np.zeros(n_actions, dtype=int)
    for t in count():
        action = select_action(state)
        raw_next_state, reward, terminated, truncated, info = train_env.step(action.item())
        row_column_index = info["matrix_position"]
        confusion_matrix[row_column_index[0], row_column_index[1]] += 1
        test.append(info["answer"])
        show[action.item()] += 1
        reward = get_reward(info["action"], info["answer"])
        if terminated:
            with open("train.log", "a") as f:
                f.write(f"{i_episode}, {show}\n")
            next_state = None
        else:
            next_state = f_p.to_tensor(raw_next_state, device=device)
        reward = torch.tensor([reward], device=device, dtype=torch.float32)
        memory.push(state, action, next_state, reward)
        sum_reward += reward.item() if reward.item() == 1 else 0
        state = next_state
        loss = optimize_model()
        if terminated:
            loss_array.append(loss)
            episode_rewards.append(sum_reward / (t + 1))
            break
    base = confusion_matrix[1, 1] + confusion_matrix[1, 0]
    episode_precision.append(
        confusion_matrix[1, 1] / base if base != 0 else 0
    )
    if i_episode > 0 and i_episode % 5 == 0:
        plot_double_graph(episode_rewards, np.array(loss_array))

plot_graph(episode_precision, show_result=True)
plot_normal_graph(loss_array, show_result=True)
torch.save(policy_net.state_dict(), "re_01_dqn_cic.pth")
train_env.close()
torch.save(policy_net.state_dict(), "re_01_dqn_cic.pth")

def plot_confusion_matrix(confusion_array, class_names=None, show_result=False):
    plt.figure(figsize=(10, 8))
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title(f"Testing...")
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(confusion_array))]
    conf_matrix = confusion_array.copy().astype(float)
    for column in range(n_actions):
        sum_column = 0
        for row in range(n_actions):
            sum_column += conf_matrix[row][column]
        if sum_column == 0:
            continue
        for row in range(n_actions):
            conf_matrix[row][column] /= sum_column
    sns.heatmap(
        conf_matrix,
        annot=True,
        cmap='Blues',
        fmt=".0%",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.grid()
    plt.pause(0.1)
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

MODEL_PATH = "re_01_dqn_cic.pth"
trained_network = DeepFlowNetwork(n_inputs=n_inputs, n_outputs=n_actions).to(device)
trained_network.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
trained_network.eval()
counts = test_df["Label"].value_counts()
confusion_array = np.zeros((n_actions, n_actions), dtype=np.int32)
metrics_dictionary = {
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1": [],
    "fpr": []
}
for i_loop in range(1):
    random.seed(i_loop)
    test_raw_state = test_env.reset()
    try:
        test_state = f_p.to_tensor(test_raw_state, device=device)
    except Exception:
        raise print(test_raw_state)
    for t in count():
        with torch.no_grad():
            test_action = trained_network(test_state).max(1).indices.view(1, 1)
        test_raw_next_state, test_reward, test_terminated, test_truncated, test_info = test_env.step(test_action.item())
        action = test_info["action"]
        answer = test_info["answer"]
        confusion_array[action, answer] += 1
        if test_terminated:
            break
        test_state = f_p.to_tensor(test_raw_next_state, device=device)
        if t % 100000 == 0:
            plot_confusion_matrix(confusion_array)
for i in range(n_actions):
    print("[", end="")
    for j in range(n_actions):
        if j == n_actions - 1:
            print(confusion_array[i, j], end="")
        else:
            print(confusion_array[i, j], end=", ")
    print("],")
plot_confusion_matrix(confusion_array, show_result=True)
