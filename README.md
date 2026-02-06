# Comparative Analysis of DRL Algorithms in IoT-Based Cyberattacks

This repository contains the code and notebooks used for performing a **comparative analysis of Deep Reinforcement Learning (DRL) algorithms** for detecting or analyzing IoT-based cyberattacks.

The aim of this project is to evaluate and compare how various DRL techniques perform in identifying, classifying, or mitigating cyber threats in IoT environments (e.g., network traffic attacks, intrusion detection, anomaly detection).

> *If you’re including this in an academic paper or thesis, link to it under “Code Availability”.*

---

##  Project Structure

| File / Folder | Description |
|---------------|-------------|
| `Preprocessing.ipynb` | Data cleaning, feature engineering, and preprocessing steps. |
| `AE_Training.ipynb` | Autoencoder or representation learning training notebook. |
| `Code.ipynb` | Main experiment notebook — model definitions + comparisons. |
| `convAE.py` | Python implementation of convolutional autoencoder (if used). |
| `requirements.txt` | (Optional) Python dependencies file. |

---

##  Background & Motivation

Cybersecurity for IoT systems is challenging due to device heterogeneity and evolving attack patterns. Traditional intrusion detection systems (IDS) often rely on static patterns and labeled datasets, which struggle with unknown threats.

DRL algorithms adapt over time, enabling IDS systems to **learn patterns dynamically** and handle evolving threats — making them a strong candidate for real-world IoT cyber defense. :contentReference[oaicite:1]{index=1}

---

##  Key Features

✔ Notebook-based experimentation (easy to reproduce)  
✔ Preprocessing pipeline for datasets  
✔ Training & evaluation of DRL models  
✔ Comparative performance analysis

---

##  Requirements

Install dependencies:

```bash
pip install -r requirements.txt
