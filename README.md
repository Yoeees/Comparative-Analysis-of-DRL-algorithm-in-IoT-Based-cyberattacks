# Semi-Supervised Deep Reinforcement Learning for IoT Anomaly Detection

## Project Overview
This project implements a semi-supervised Deep Reinforcement Learning (DRL) framework for detecting cyberattacks in Internet of Things (IoT) networks. The framework integrates a Convolutional Autoencoder (ConvAE) with multiple DRL algorithms to enable anomaly detection under limited labeled data conditions.

The study focuses on a controlled comparison between off-policy and on-policy DRL approaches for time-series IoT traffic, emphasizing short-burst and sequential attack patterns.

## Dataset
Dataset Name: Edge-IIoTset

Edge-IIoTset is a comprehensive IoT/IIoT cybersecurity dataset generated from a realistic edge–cloud testbed environment. It contains packet-level network traffic collected from heterogeneous IoT devices and services under both benign and malicious conditions.

Key dataset properties:
- IoT-specific traffic and protocols
- Multiple attack categories, including DDoS, MITM, scanning, injection, and malware
- Highly imbalanced class distribution
- Suitable for time-series and window-based analysis
- Publicly available and widely used in IoT security research

## Objectives

### General Objective
To conduct a comparative analysis of off-policy and on-policy Deep Reinforcement Learning algorithms for semi-supervised IoT anomaly detection using autoencoder-based intrinsic rewards on the Edge-IIoTset dataset.

### Specific Objectives
- Implement Deep Q-Network (DQN), Quantile Regression DQN (QR-DQN), Proximal Policy Optimization (PPO), and Recurrent PPO (LSTM) for anomaly detection.
- Integrate a Convolutional Autoencoder to generate latent representations and reconstruction error signals.
- Apply sliding-window time-series modeling to capture short-burst and temporal attack behaviors.
- Evaluate models using standard classification and detection metrics such as Precision, Recall, F1-score, AUC-ROC, and False Positive Rate.
- Analyze the impact of off-policy versus on-policy learning paradigms in semi-supervised IoT security settings.

## Methodology Summary
- IoT network traffic is transformed into overlapping sliding windows to preserve temporal dependencies.
- A Convolutional Autoencoder is pretrained on normal traffic to learn compact representations and reconstruction errors.
- Reconstruction error is used as an intrinsic reward signal, while a small subset of labeled anomalies provides sparse extrinsic rewards.
- A custom OpenAI Gym environment simulates sequential IoT traffic for DRL training.
- DRL agents are trained and evaluated using Stable-Baselines3 under identical conditions for fair comparison.

## Implemented Algorithms
- Deep Q-Network (DQN)
- Quantile Regression DQN (QR-DQN)
- Proximal Policy Optimization (PPO)
- Recurrent Proximal Policy Optimization (PPO with LSTM)

## Evaluation Metrics
- Precision
- Recall
- F1-score
- AUC-ROC
- False Positive Rate

## Requirements
- Python 3.8 or later
- PyTorch
- Stable-Baselines3
- OpenAI Gym
- NumPy
- Pandas
- Scikit-learn

## Notes
This repository is intended for academic and research use. The implementation prioritizes reproducibility and controlled experimentation for IoT anomaly detection using Deep Reinforcement Learning.
