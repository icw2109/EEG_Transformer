# -*- coding: utf-8 -*-
"""neural_deep_learning_beta_two.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1nXLwE9EkFZyhX4p2Y7ueFXc-mUObYtZA
"""



!pip install torch_geometric
!pip install torcheeg
!pip install torchvision

import random
import numpy as np
import torch
from torcheeg.datasets import SEEDIVFeatureDataset
from torcheeg import transforms
from torcheeg.datasets.constants import SEED_IV_CHANNEL_LOCATION_DICT, SEED_IV_ADJACENCY_MATRIX
from torch import nn
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn import svm
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pickle
from time import time
from torcheeg.models import FBCNet, GRU, FBCCNN, EEGNet, LSTM, DGCNN, ViT
from torch.optim import Adam
import torch.nn.functional as F
import os
from sklearn.model_selection import train_test_split
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

print(os.listdir('/content/drive/MyDrive/archive'))

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0.0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def valid(dataloader, model, loss_fn):
    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            X = batch[0].to(device)
            y = batch[1].to(device)
            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    val_loss /= len(dataloader)
    correct /= len(dataloader.dataset)
    return correct, val_loss

def valid_with_report(dataloader, model, loss_fn, model_name, idx):
    model.eval()
    val_loss, correct = 0, 0
    all_predictions = []
    all_targets = []
    output_path = './output'
    with torch.no_grad():
        for batch in dataloader:
            X = batch[0].to(device)
            y = batch[1].to(device)
            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            all_predictions.extend(pred.argmax(1).cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    val_loss /= len(dataloader)
    correct /= len(dataloader.dataset)

    cm = confusion_matrix(all_targets, all_predictions)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, cmap='Blues', fmt='.2f')
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig('{}/normalized_confusion_matrix_{}_{}.png'.format(output_path, model_name, idx))
    plt.show()

    classification_report_str = classification_report(all_targets, all_predictions)
    print("{} {} Classification Report:\n".format(model_name, idx))
    print(classification_report_str)



def import_data1():
    dataset = SEEDIVFeatureDataset(
        root_path='/content/drive/MyDrive/archive/eeg_feature_smooth',
        feature=['de_movingAve', 'de_LDS', 'psd_movingAve', 'psd_LDS'],
        online_transform=transforms.Compose([transforms.ToTensor()]),
        label_transform=transforms.Select('emotion'),
        num_worker=4,
        io_size=167772160
    )
    return dataset

def import_data2():
    dataset = SEEDIVFeatureDataset(
        root_path='/content/drive/MyDrive/archive/eeg_feature_smooth',
        feature=['de_movingAve', 'de_LDS', 'psd_movingAve', 'psd_LDS'],
        online_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.To2d()
        ]),
        label_transform=transforms.Select('emotion'),
        num_worker=4,
        io_size=167772160
    )
    return dataset

def import_data3():
    dataset = SEEDIVFeatureDataset(
        root_path='/content/drive/MyDrive/archive/eeg_feature_smooth',
        feature=['de_movingAve', 'de_LDS', 'psd_movingAve', 'psd_LDS'],
        online_transform=transforms.Compose([
            transforms.ToGrid(SEED_IV_CHANNEL_LOCATION_DICT),
            transforms.ToTensor()
        ]),
        label_transform=transforms.Compose([
            transforms.Select('emotion'),
            transforms.Lambda(lambda x: x)
        ]),
        num_worker=4,
        io_size=167772160
    )
    return dataset

def import_data4():
    dataset = SEEDIVFeatureDataset(
        root_path='/content/drive/MyDrive/archive/eeg_feature_smooth',
        feature=['de_movingAve', 'de_LDS', 'psd_movingAve', 'psd_LDS'],
        online_transform=transforms.Compose([
            transforms.pyg.ToG(SEED_IV_ADJACENCY_MATRIX)
        ]),
        label_transform=transforms.Compose([
            transforms.Select('emotion'),
            transforms.Lambda(lambda x: x)
        ]),
        num_worker=4,
        io_size=167772160
    )
    return dataset

def get_dataloaders(dataset, batch_size):
    num_training = int(len(dataset) * 0.7)
    num_val = int(len(dataset) * 0.15)
    num_test = len(dataset) - num_val - num_training

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [num_training, num_val, num_test], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def import_GRU():
    model = GRU(num_electrodes=62, hid_channels=64, num_classes=4).to(device)
    data = import_data1()
    train_loader, val_loader, test_loader = get_dataloaders(data, 64)
    return model, train_loader, val_loader, test_loader

def import_LSTM():
    model = LSTM(num_electrodes=62, hid_channels=64, num_classes=4).to(device)
    data = import_data1()
    train_loader, val_loader, test_loader = get_dataloaders(data, 64)
    return model, train_loader, val_loader, test_loader

class EEGTransformer(nn.Module):
    def __init__(self, num_electrodes, num_classes, seq_length, d_model=64, nhead=4, num_encoder_layers=3, dim_feedforward=256, dropout=0.1):
        super(EEGTransformer, self).__init__()
        self.num_electrodes = num_electrodes
        self.seq_length = seq_length
        self.embedding = nn.Linear(num_electrodes, d_model)
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.output_layer = nn.Linear(d_model, num_classes)

    def forward(self, src):
        src = src.view(-1, self.seq_length, self.num_electrodes)
        src = self.embedding(src)
        src = self.dropout(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=1)
        return self.output_layer(output)

def import_EEGTransformer():
    model = EEGTransformer(num_electrodes=62, num_classes=4, seq_length=20, d_model=64, nhead=4, num_encoder_layers=3).to(device)
    data = import_data1()
    train_loader, val_loader, test_loader = get_dataloaders(data, 64)
    return model, train_loader, val_loader, test_loader

def train_fun(seed, model, model_name, epochs, train_loader, val_loader, test_loader):
    set_random_seed(seed)
    optimizer = Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    final_test_acc = 0.0
    best_epoch = 0
    train_loss_list = []
    train_acc_list = []
    val_acc_list = []
    train_times = []

    # Ensure the output directory exists
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)

    for t in range(epochs):
        s_time = time()
        train_loss = train(train_loader, model, loss_fn, optimizer)
        train_times.append(time() - s_time)
        train_acc, _ = valid(train_loader, model, loss_fn)
        val_acc, val_loss = valid(val_loader, model, loss_fn)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            test_acc, _ = valid(test_loader, model, loss_fn)
            final_test_acc = test_acc
        if (t + 1) % 10 == 0:
            log_format = (
                "Epoch {}: loss={:.4f}, train_acc={:.4f}, val_acc={:.4f}, final_test_acc={:.4f}"
            )
            print(log_format.format(t + 1, train_loss, train_acc, val_acc, final_test_acc))
    print(
        "Best Epoch {}, final test acc {:.4f}".format(
            best_epoch, final_test_acc
        )
    )
    print("Done!")

    torch.save(model.state_dict(), '{}/last_{}_weights_trial{}.pth'.format(output_dir, model_name, seed))
    valid_with_report(test_loader, model, loss_fn, model_name, seed)

    return final_test_acc, sum(train_times) / len(train_times), [train_loss_list, train_acc_list, val_acc_list]


# Statistical functions
def mean_of_n_values(arr, smoth):
    arr = arr.reshape(arr.shape[0], -1, smoth)
    mean = arr.mean(axis=2)
    return mean.reshape(mean.shape[0], -1)

def boxplot(accs, output_path, name, model_type):
    fig, ax = plt.subplots(figsize=(14, 10), dpi=80)
    ax.set_title('Test Accuracy of different trial models')
    ax.boxplot(accs)
    ax.set_xticklabels([model_type])
    plt.savefig('{}/boxplot_{}_{}.png'.format(output_path, name, model_type))
    plt.show()

def acc_loss_plot(results, epochs, smoth, num_trials, output_path, feat_type):
    results = np.array(results)

    plt.figure(figsize=(14, 10), dpi=80)

    x = np.arange(0, epochs, smoth)

    loss_train_results = mean_of_n_values(results[:, 0, :], smoth)
    acc_train_results = mean_of_n_values(results[:, 1, :], smoth)
    acc_valid_results = mean_of_n_values(results[:, 2, :], smoth)

    plt.plot(x, np.mean(acc_train_results, axis=0), label='Train Accuracy')
    plt.fill_between(x, np.min(acc_train_results, axis=0), np.max(acc_train_results, axis=0), alpha=0.2, label='Train Accuracy Noise')
    plt.plot(x, np.mean(acc_valid_results, axis=0), label='Valid Accuracy')
    plt.fill_between(x, np.min(acc_valid_results, axis=0), np.max(acc_valid_results, axis=0), alpha=0.2, label='Valid Accuracy Noise')
    plt.title("Train plot and valid plot over {:4d} models plotted the mean with min noise and max noise smoothed every {:4d} epochs".format(num_trials, smoth))
    plt.legend()
    plt.savefig('{}/accuracy_plot_valid_train_{}.png'.format(output_path, feat_type))
    plt.show()

    plt.figure(figsize=(14, 10), dpi=80)

    plt.plot(x, np.mean(loss_train_results, axis=0), label='Train Loss')
    plt.fill_between(x, np.min(loss_train_results, axis=0), np.max(loss_train_results, axis=0), alpha=0.2, label='Train Loss Noise')
    plt.title("Loss plot over {:4d} models plotted the mean with min noise and max noise smoothed every {:4d} epochs".format(num_trials, smoth))
    plt.legend()
    plt.savefig('{}/loss_plot_train_{}.png'.format(output_path, feat_type))
    plt.show()

def stats(model_name, num_trials):
    boxplot(list_last_accs, './output', "Test_Accuracy", model_name)
    acc_loss_plot(list_stat_list, epochs, 1, num_trials, './output', model_name)

def get_stats(array, conf_interval=False):
    import torch
    import math
    from scipy.stats import t

    eps = 1e-9
    array = torch.Tensor(array)
    std, mean = torch.std_mean(array)
    std = std.item()
    mean = mean.item()
    center = mean

    if conf_interval:
        n = array.size(0)
        se = std / (math.sqrt(n) + eps)
        t_value = t.ppf(0.975, df=n - 1)
        err_bound = t_value * se
    else:
        err_bound = std

    return center, err_bound

def import_model_and_data(idx):
    if models_names[idx] == "GRU":
        return import_GRU()
    elif models_names[idx] == "LSTM":
        return import_LSTM()
    elif models_names[idx] == "EEGTransformer":
        return import_EEGTransformer()
    else:
        raise ValueError("Model name not recognized")

models_names = ["GRU", "LSTM", "EEGTransformer"]
num_trials = 5
batch_size = 64
epochs = 10

list_accs = []
list_stat_list = []
list_last_accs = []
models_times = []

for idx, model_name in enumerate(models_names):
    accs = []
    best_acc = -1
    stat_list = []
    time_taken_total = 0

    for trial in range(num_trials):
        print(f"Trial {trial + 1}/{num_trials}")
        set_random_seed(trial)
        model, train_loader, val_loader, test_loader = import_model_and_data(idx)
        acc, time_taken, stat = train_fun(trial, model, model_name, epochs, train_loader, val_loader, test_loader)
        accs.append(acc)

        if best_acc < acc:
            best_acc = acc

        stat_list.append(stat)
        time_taken_total += time_taken

    list_last_accs.append(best_acc)
    list_stat_list.append(stat_list)
    list_accs.append(accs)
    models_times.append(time_taken_total / num_trials)

    mean, err_bd = get_stats(accs)
    print(f"{model_name} mean acc: {mean:.4f}, error bound: {err_bd:.4f}, time taken {time_taken_total / num_trials:.4f}")

with open('list_last_accs.pkl', 'wb') as file:
    pickle.dump(list_last_accs, file)
with open('list_stat_list.pkl', 'wb') as file:
    pickle.dump(list_stat_list, file)
with open('list_accs.pkl', 'wb') as file:
    pickle.dump(list_accs, file)
with open('models_times.pkl', 'wb') as file:
    pickle.dump(models_times, file)  ## Epochs to low

class EEGEnvironment:
    def __init__(self, eeg_data):
        self.eeg_data = eeg_data
        self.current_step = 0
        self.num_steps = len(eeg_data)

    def reset(self):
        self.current_step = 0
        return self.eeg_data[self.current_step]

    def step(self, action):
        self.current_step += 1
        if self.current_step < self.num_steps:
            next_state = self.eeg_data[self.current_step]
            reward = self.calculate_reward(action)
            done = False
        else:
            next_state = None
            reward = 0
            done = True
        return next_state, reward, done

    def calculate_reward(self, action):
        return random.random()

    def sample_action(self):
        return random.choice([0, 1, 2, 3])


class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.memory = []
        self.model = DQNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return np.argmax(q_values.numpy())

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                with torch.no_grad():
                    target = reward + self.gamma * torch.max(self.model(next_state)).item()
            state = torch.FloatTensor(state).unsqueeze(0)
            target_f = self.model(state)
            target_f[0][action] = target
            self.optimizer.zero_grad()
            output = self.model(state)
            loss = self.loss_fn(output, target_f)
            loss.backward()
            self.optimizer.step()

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

def train_dqn(agent, environment, num_episodes=1000, batch_size=32):
    for episode in range(num_episodes):
        state = environment.reset()
        total_reward = 0
        while True:
            action = agent.act(state)
            next_state, reward, done = environment.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        print(f'Episode {episode + 1}, Total Reward: {total_reward}')

eeg_data = np.random.rand(100, 62)
environment = EEGEnvironment(eeg_data)
agent = DQNAgent(state_size=62, action_size=4)  # Adjust state_size to match the actual state representation
train_dqn(agent, environment)
