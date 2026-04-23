# Reinforcement Learning Trading Bot for EUR/USD Forex

A quantitative trading system that trains a Deep Q-Network (DQN) agent to trade the EUR/USD currency pair. The agent learns optimal buy/sell/hold decisions directly from historical price data through reinforcement learning, receiving rewards based on trading performance and adapting its policy through experience replay.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [DQN Architecture](#dqn-architecture)
- [State Space](#state-space)
- [Action Space](#action-space)
- [Reward Function](#reward-function)
- [Results](#results)
- [License](#license)

---

## Overview

This project demonstrates the application of deep reinforcement learning to financial trading. Unlike supervised or unsupervised approaches, reinforcement learning allows the agent to discover optimal trading strategies through trial and error, learning from its own trading outcomes rather than from pre-labelled data.

| Parameter   | Value                      |
| ----------- | -------------------------- |
| Asset       | EUR/USD (Euro / US Dollar) |
| Data Source | Yahoo Finance via yfinance |
| Period      | 2010-01-01 to 2024-12-31   |
| Frequency   | Daily OHLCV                |
| Algorithm   | Deep Q-Network (DQN)       |

---

## Pipeline Architecture

```
Raw OHLCV Data (EUR/USD)
      |
      v
Feature Engineering / State Representation
      |
      v
DQN Agent Training
      |
      +----> Experience Replay Buffer
      |
      +----> Target Network (periodic updates)
      |
      +----> Epsilon-Greedy Exploration
      |
      v
Learned Policy (Q-Network)
      |
      v
Trading Signal Generation ----> Backtest
```

---

## Requirements

- Python 3.9+
- NumPy
- Pandas
- Matplotlib
- TensorFlow / Keras
- yfinance

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/abrar2030/Forex_Trading_Bot.git
cd Forex_Trading_Bot
```

2. Install dependencies (executed automatically in the notebook):

```bash
pip install numpy pandas matplotlib tensorflow yfinance
```

3. Open the Jupyter Notebook:

```bash
jupyter notebook Forex_Trading_Bot.ipynb
```

---

## Usage

Run all cells in the notebook sequentially. The pipeline is fully self-contained and will:

1. Download EUR/USD historical data from Yahoo Finance
2. Build a custom trading environment compatible with reinforcement learning
3. Define the state space, action space, and reward function
4. Train a DQN agent with experience replay and target network stabilisation
5. Evaluate the trained agent on out-of-sample data
6. Backtest the learned trading policy against benchmarks
7. Visualise training progress, Q-value evolution, and equity curves

---

## Methodology

### Environment Design

The trading environment follows the OpenAI Gym interface:

- **reset()**: Initialises the episode at a random starting point
- **step(action)**: Executes the action, returns next state, reward, and done flag
- **render()**: Visualises the current state and agent position

### Training Process

1. Agent observes the current market state
2. Selects an action using epsilon-greedy policy
3. Receives reward based on trading outcome
4. Stores transition in replay buffer
5. Samples mini-batches from replay buffer for training
6. Updates Q-network weights via gradient descent
7. Periodically syncs target network with policy network
8. Decays exploration rate epsilon over time

---

## DQN Architecture

| Component     | Description                                              |
| ------------- | -------------------------------------------------------- |
| Input Layer   | Normalised state vector                                  |
| Hidden Layers | Fully connected dense layers with ReLU activation        |
| Output Layer  | Q-value for each action (Buy, Sell, Hold)                |
| Optimiser     | Adam                                                     |
| Loss Function | Mean Squared Error between predicted and target Q-values |

---

## State Space

The agent observes a feature vector derived from recent market data, which may include:

- Normalised price changes over multiple lookback windows
- Technical indicators (RSI, MACD, moving averages)
- Current position status (flat, long, short)
- Recent return history
- Volatility measures

---

## Action Space

| Action   | Description                            |
| -------- | -------------------------------------- |
| Buy (0)  | Enter or maintain long position        |
| Sell (1) | Enter or maintain short position       |
| Hold (2) | Maintain current position or stay flat |

---

## Reward Function

The agent receives rewards based on:

- **Realised P&L**: Profit or loss from closed trades
- **Unrealised P&L**: Mark-to-market on open positions
- **Risk-adjusted returns**: Returns penalised for high volatility exposure
- **Transaction costs**: Small penalty for each trade to discourage excessive turnover

---

## Results

The notebook produces:

- Training reward curve showing learning progress
- Q-value heatmaps for different market states
- Epsilon decay schedule
- Action distribution analysis
- Backtest equity curve vs buy-and-hold benchmark
- Trade-level statistics (win rate, average profit/loss, max consecutive wins/losses)
- Drawdown analysis

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
