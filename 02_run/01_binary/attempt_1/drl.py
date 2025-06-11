from sklearn.preprocessing import LabelEncoder
from glob import glob
from tqdm import tqdm
import pandas as pd
import flow_package as f_p
import pandas as pd
from glob import glob
import random
import os
import torch
import time

from collections import deque, namedtuple
from itertools import count

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
import torch.optim as optim

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))



is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
CONST = f_p.Const()

TRAIN_DIR = "data_cicids2017/1_sampling"

files_path = glob(f"{TRAIN_DIR}/bi_*.csv")
print(files_path)

# データ準備の計測
start_time = time.time()
df = pd.DataFrame()
for file_path in tqdm(files_path):
    df_tmp = pd.read_csv(file_path)
    df_tmp = df_tmp.dropna()
    df_tmp = df_tmp.drop_duplicates()
    df = pd.concat([df, df_tmp], axis=0)
print(f"データ準備時間: {time.time() - start_time:.2f}秒")

from sklearn.model_selection import train_test_split

buf, test_df = train_test_split(df, test_size=0.3, random_state=50)
train_df, _ = train_test_split(buf, test_size=0.2, random_state=42)
from flow_package.binary_flow_env import BinaryFlowEnv, InputType

train_input = InputType(
    input_features=train_df.drop(columns=["Label"]),
    input_labels=train_df["Label"],
    reward_list=[1.0, -1.0],
    
)

test_input = InputType(
    input_features=test_df.drop(columns=["Label"]),
    input_labels=test_df["Label"],
    reward_list=[1.0, -1.0],
    type_env="test"
)

train_env = BinaryFlowEnv(train_input)
test_env = BinaryFlowEnv(test_input)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Transaction = namedtuple('Transaction', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        # self.capacity = capacity
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

def plot_graph(data: list, show_result=False, only_save=False):
    plt.figure(figsize=(15,5))
    # durations_t = torch.tensor(episode_durations, dtype=torch.float)

    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")
    
    means = moving_average(data, 50)

    plt.xlabel("Episode")
    plt.ylabel("ratio")
    plt.plot(data, color="gold")
    plt.plot(means, color="red")
    plt.grid()

    if only_save:
        plt.savefig(f"result/binary/precision.png")
        plt.close()
        return

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

PORT_DIM = 8
class DeepFlowNetwork(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(DeepFlowNetwork, self).__init__()

        self.protocol_embedding = nn.Embedding(256, 8) # -> 8
        self.port_embedding = nn.Embedding(65536, PORT_DIM) # -> 8
        # other inputs are not embedding: n_inputs - 2

        # all inputs: n_inputs - 2 + 8 + 8 = n_inputs + 14
        n_inputs = n_inputs + 6 + PORT_DIM

        self.fc1 = nn.Linear(n_inputs, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, n_outputs)

    def forward(self, x):
        port_emb = self.port_embedding(x[0])
        protocol_emb = self.protocol_embedding(x[1])

        # print(port_emb.shape, protocol_emb.shape, x[2].shape)

        renew = torch.cat([port_emb, protocol_emb, x[2]], dim=1)

        renew = F.relu(self.fc1(renew))
        renew = F.relu(self.fc2(renew))
        return self.fc3(renew)


BATCH_SIZE = 256
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 100_000
TAU = 0.005
LR = 1e-7
n_actions = train_env.action_space.n
n_inputs = train_env.observation_space.shape[0]

# print(f"n_inputs: {n_inputs}, n_actions: {n_actions}")

state = train_env.reset()
# print(info)

policy_net = DeepFlowNetwork(n_inputs, n_actions).to(device)
target_net = DeepFlowNetwork(n_inputs, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.SGD(policy_net.parameters(), lr=LR)
steps_done = 0

memory = ReplayMemory(10000)
episode_rewards = []
episode_precision = []

def select_action(state_tensor: torch.Tensor):
    # print(f"state_tensor: {state_tensor[0,1]}")
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            # print(policy_net(state_tensor))
            # print(policy_net(state_tensor).max(1))
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
    # non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    # print(batch.state[0])
    state_batch_port = torch.cat([s[0] for s in batch.state])
    state_batch_protocol = torch.cat([s[1] for s in batch.state])
    state_batch_other = torch.cat([s[2] for s in batch.state])

    state_batch = [state_batch_port, state_batch_protocol, state_batch_other]
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    # non_final_next_states も同様に処理
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

def test_model():
    MODEL_PATH = "re_01_dqn_cic.pth"

    # load the model
    trained_network = DeepFlowNetwork(n_inputs=n_inputs, n_outputs=n_actions).to(device)
    trained_network.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    trained_network.eval()

    # test the model

    confusion_array = np.zeros((2, 2), dtype=np.int32)
    metrics_dictionary = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "fpr": []
    }

    for i_loop in range(2):
        random.seed(i_loop)
        test_raw_state = test_env.reset()
        try:
            test_state = f_p.to_tensor(test_raw_state, device)
        except:
            raise print(test_raw_state)
        for t in count():
            with torch.no_grad():
                test_action = trained_network(test_state).max(1).indices.view(1, 1)

            test_raw_next_state, test_reward, test_terminated, test_truncated, test_info = test_env.step(test_action.item())
            # print(test_info)
            # calculate confusion matrix
            raw = 0 if test_reward == 1 else 1

            # test_info = (row, column) means confusion matrix index
            index = test_info["matrix_position"]
            confusion_array[index[0], index[1]] += 1

            # print(index)

            if test_terminated:
                break

            # make next state tensor and update state
            #test_state = torch.tensor(test_raw_next_state, device=device, dtype=torch.float32).unsqueeze(0)
            test_state = f_p.to_tensor(test_raw_next_state, device)

        # calculate metrics
        tp = confusion_array[1, 1]
        tn = confusion_array[0, 0]
        fp = confusion_array[1, 0]
        fn = confusion_array[0, 1]
        print(f"{i_loop + 1:5}, {tp:7}, {tn:7}, {fp:7}, {fn:7}")

        accuracy, precision, recall, f1, fpr = calculate_metrics(tp, tn, fp, fn)
        metrics_dictionary["accuracy"].append(accuracy)
        metrics_dictionary["precision"].append(precision)
        metrics_dictionary["recall"].append(recall)
        metrics_dictionary["f1"].append(f1)
        metrics_dictionary["fpr"].append(fpr)
        # print(accuracy, precision, recall, f1, fpr)


        # if i_loop % 50 == 0:
        #     print(f"{i_loop + 1:5}, {tp:7}, {tn:7}, {fp:7}, {fn:7}")

    # plot metrics
    print(f" accuracy: {metrics_dictionary['accuracy'][-1]}")
    print(f"precision: {metrics_dictionary['precision'][-1]}")
    print(f"  recall : {metrics_dictionary['recall'][-1]}")
    print(f"    f1   : {metrics_dictionary['f1'][-1]}")
    print(f"   fpr   : {metrics_dictionary['fpr'][-1]}")

num_episodes = 1_000

for i_episode in range(num_episodes):
    # episode_start = time.time()
    # Initialize the environment and state
    random.seed(i_episode)
    sum_reward = 0
    confusion_matrix = np.zeros((2,2), dtype=int)

    initial_state = train_env.reset()
    # state = torch.tensor(initial_state, device=device, dtype=torch.float32).unsqueeze(0)
    state = f_p.to_tensor(initial_state, device)

    for t in count():
        step_start = time.time()
        # select action
        action = select_action(state)

        # print(action)
        # calculate next state
        raw_next_state, reward, terminated, truncated, info = train_env.step(action.item())
        row_column_index = info["matrix_position"]
        # print(info)
        confusion_matrix[row_column_index[0], row_column_index[1]] += 1
        # print(info)

        # to tensor
        # if terminated and t > 1000:
        if terminated:
            next_state = None
        else:
            # next_state = torch.tensor(raw_next_state, device=device, dtype=torch.float32).unsqueeze(0)
            next_state = f_p.to_tensor(raw_next_state, device)
        reward = torch.tensor([reward], device=device, dtype=torch.float32)

        # store the transition in memory
        memory.push(state, action, next_state, reward)
        sum_reward += reward.item() if reward.item() == 1 else 0

        # move to the next state
        state = next_state

        # optimize the model
        # print("optimize_model")
        optimize_start = time.time()
        optimize_model()
        optimize_time = time.time() - optimize_start
        step_time = time.time() - step_start
        # if t % 100 == 0:
            # print(f"  エピソード{i_episode} ステップ{t} step_time: {step_time:.4f}秒, optimize_time: {optimize_time:.4f}秒")

        if terminated:
            # if t > 1000:
            episode_rewards.append(sum_reward / (t + 1))
            break

    # do after the episode
    # episode_rewards.append(sum_reward)
    base = confusion_matrix[1, 1] + confusion_matrix[1, 0]
    episode_precision.append(
        confusion_matrix[1, 1] / base if base != 0 else 0
    )
    # print(i_episode)
    if i_episode > 0 and i_episode % 50 == 0:
        plot_graph(episode_precision, only_save=True)
        print(f"episode: {i_episode:4d}, precision: {episode_precision[-1]:.4f}")
    if i_episode > 0 and i_episode % 100 == 0:
        torch.save(policy_net.state_dict(), f"re_01_dqn_cic.pth")
        print("="*100)
        test_model()
        print("="*100)
    # episode_time = time.time() - episode_start
    # print(f"エピソード{i_episode} 終了: {episode_time:.2f}秒")

# complete the episode
plot_graph(episode_precision, show_result=True, only_save=True)
torch.save(policy_net.state_dict(), "re_01_dqn_cic.pth")  # save the model

train_env.close()