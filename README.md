# 🍔 Food Classifier

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-1.14%2B-orange)
![GitHub Actions](https://img.shields.io/badge/CI-CD-GitHub%20Actions-brightgreen)

---

## Overview

Food Classifier is a **image classification project** built with PyTorch and FastAPI. It classifies food images leveraging **pretrained ResNet models**, and provides rich visualization, reporting, and API integration for real-time predictions.  

The project is designed to showcase **end-to-end ML engineering best practices**, including data preprocessing, training, evaluation, visualization, and CI/CD.

---

## Features

### 1️⃣ Model & Training

- Fine-tuned **ResNet models** for food classification using [Food-101](https://www.kaggle.com/datasets/dansbecker/food-101) and custom scraped datasets.
- Supports **TensorBoard logging** for loss, accuracy, and other metrics.
- Provides **Grad-CAM heatmaps** for model explainability, highlighting which parts of an image influence predictions.
- Generates detailed **metrics reports** including per-class accuracy, precision, recall, F1, and **top confusions** via matplotlib.

### 2️⃣ Data Preparation

- Unified multiple datasets (Food-101 + custom scraped images).
- Includes a **scraper module** using Google Scraper to extend datasets automatically.
- Standardized preprocessing pipeline with **train/validation/test splits** and augmentation for robust model training.

### 3️⃣ Prediction & API

- `predict.py` supports **single-image prediction** from local directories.
- Generates **Grad-CAM visualizations** alongside predictions.
- FastAPI-based API exposing **predict endpoint** for real-time image classification.

### 4️⃣ Testing & CI/CD

- Comprehensive **unit tests** for utility functions and **integration tests** for evaluation and prediction pipelines.
- **GitHub Actions workflow** for automatic CI:
  - Linting with Black & Flake8
  - Running unit & integration tests
- Ensures code quality, reproducibility, and maintainability.

---
