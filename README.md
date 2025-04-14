# EEG Deep Learning and Reinforcement Learning Project

## Overview
This repository includes implementations of deep learning models (GRU, LSTM, EEGTransformer) for emotion recognition using EEG data (SEED IV Dataset) and an EEG-based reinforcement learning environment utilizing a Deep Q-Network (DQN).

### EEGTransformer
The EEGTransformer is a Transformer-based neural network specifically tailored to EEG data. EEG data consists of temporal sequences recorded across multiple electrodes. In the EEGTransformer:

- **Input Handling:** EEG signals from multiple electrodes form sequences over time, organized into tensors representing time steps and electrodes.
- **Embedding Layer:** Each time-step’s EEG electrode signals are linearly projected into a higher-dimensional embedding space, capturing rich feature representations.
- **Transformer Encoder Layers:** Embedded data is fed into Transformer encoder layers that utilize multi-head self-attention mechanisms. These attention mechanisms allow the model to dynamically focus on different parts of the EEG sequence, effectively capturing temporal dependencies and spatial correlations across electrodes.
- **Output and Classification:** The output of the Transformer encoder is averaged across time steps, resulting in a fixed-size vector used for emotion classification.

This methodology is useful as it leverages attention mechanisms to automatically learn significant temporal and spatial relationships within EEG data, improving the accuracy of emotion recognition tasks compared to traditional recurrent networks.

### Deep Q-Network (DQN) Agent
The Deep Q-Network (DQN) agent operates within an EEG-based reinforcement learning environment:

- **Environment Interaction:** The agent receives EEG data states representing neural activity at a specific time step.
- **Action Selection:** The agent selects actions (emotions) based on learned policies. Decisions are made using an epsilon-greedy strategy, balancing exploration of new actions and exploitation of known rewards.
- **Learning and Optimization:** The agent stores experiences (state, action, reward, next state) in memory and periodically samples from this memory to learn optimal policies. The DQN uses a neural network to approximate Q-values, estimating future rewards and updating its parameters through backpropagation and gradient descent.
- **Goal:** The agent continuously improves its ability to predict correct emotional states from EEG signals by maximizing cumulative rewards over time.

## Model Structure

```
.
├── models
│   ├── GRU
│   ├── LSTM
│   └── EEGTransformer
├── reinforcement_learning
│   ├── environment.py
│   └── agent.py
├── data
│   └── SEED_IV_dataset
├── output
│   ├── confusion_matrices
│   ├── model_weights
│   └── training_plots
├── requirements.txt
└── README.md
```


## Dependencies
- Python 3.8+
- PyTorch
- torch-geometric
- torcheeg
- torchvision
- numpy
- matplotlib
- sklearn
- seaborn

## Dataset

The SEED IV EEG dataset is used:
- Download from: [SEED IV Dataset](https://bcmi.sjtu.edu.cn/~seed/seed-iv.html)
- Place dataset files in `data/SEED_IV_dataset`.

## Models Implemented

### Deep Learning Models:
- **GRU (Gated Recurrent Units)**
- **LSTM (Long Short-Term Memory)**
- **EEGTransformer (Transformer-based model)**

### Reinforcement Learning:
- **Deep Q-Network (DQN)** for EEG data-driven environment.

## Usage

### Training Deep Learning Models

Run training script:
```bash
python train_models.py
```

### Reinforcement Learning

Train DQN Agent:
```bash
python train_dqn.py
```

## Results

Trained models and plots including confusion matrices and accuracy/loss curves are stored under `output/`. Outputs are tested to be compared against baseline GRU and LSTM models.


---

**Developed by [Your Name] - [Your GitHub/Portfolio URL]**

