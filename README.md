# Sentiment-Classification
Sentiment Analysis in social media
# Sentiment Analysis of Comments

This repository contains a Jupyter Notebook for performing sentiment analysis on a dataset of comments. The notebook splits comments into positive and negative categories using a machine learning approach.

## Project Overview

The notebook is designed to preprocess comment data, train a machine learning model, and evaluate its performance in classifying comments into positive, negative, and neutral categories.

## Key Features

- **Data Preprocessing**: Cleaning and preparing the data for training.
- **Model Training**: Using a machine learning algorithm to train a sentiment analysis model.
- **Model Evaluation**: Evaluating the model's performance using various metrics.

## How to Use

Clone the repository:
    ```bash
    git clone https://github.com/Alihasan520/Sentiment-Classification.git
    ```

## Notebook Contents

- **Data Loading**: Load the dataset of comments.
- **Data Exploration**: Explore the dataset to understand its structure and content.
- **Data Preprocessing**: Clean the data, handle missing values, and prepare it for model training.
- **Model Training**: Train a machine learning model on the preprocessed data.
- **Model Evaluation**: Evaluate the model's performance using precision, recall, F1-score, and confusion matrix.

## Dependencies

- Python 3.x
- Jupyter Notebook
- pandas
- numpy
- scikit-learn

## Results

The model's performance is evaluated on both training and test datasets. Below are the key metrics:

### Training Set
- **Accuracy**: 0.71
- **Precision, Recall, F1-Score**:
    - Positive: 0.97, 0.20, 0.34
    - Neutral: 0.91, 0.62, 0.74
    - Negative: 0.62, 0.99, 0.76

### Test Set
- **Accuracy**: 0.62
- **Precision, Recall, F1-Score**:
    - Positive: 0.90, 0.08, 0.14
    - Neutral: 0.80, 0.47, 0.59
    - Negative: 0.56, 0.96, 0.70

## Confusion Matrix (Test Set)
