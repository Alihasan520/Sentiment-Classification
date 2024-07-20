# Sentiment Analysis of Comments

This repository contains a Jupyter Notebook for performing sentiment analysis on a dataset of comments. The notebook splits comments into positive and negative categories using a machine learning approach.

## Project Overview

The notebook is designed to preprocess comment data, train a machine learning model, and evaluate its performance in classifying comments into positive, negative, and neutral categories.

## Key Features

- **Data Preprocessing**: Cleaning and preparing the data for training.
- **Model Training**: Using a machine learning algorithm to train a sentiment analysis model.
- **Model Evaluation**: Evaluating the model's performance using various metrics.
- **Label Encoding**: Converting textual sentiment labels to numeric using `LabelEncoder`.

## Notebook Contents

1. **Data Loading**: Load the dataset of comments.
2. **Data Exploration**: Explore the dataset to understand its structure and content.
3. **Data Preprocessing**: Clean the data, handle missing values, and prepare it for model training.
4. **Label Encoding**: Convert textual sentiment labels to numeric values.
    ```python
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['sentiment'])
    ```
5. **Model Training**: Train a machine learning model on the preprocessed data.
6. **Model Evaluation**: Evaluate the model's performance using precision, recall, F1-score, and confusion matrix.

## Results

The model's performance is evaluated on both training and test datasets. Below are the key metrics:

### Training Set
- **Accuracy**: 0.90
- **Precision, Recall, F1-Score**:
    - Positive: 0.89, 0.75, 0.82
    - Neutral: 0.90, 0.91, 0.90
    - Negative: 0.90, 0.95, 0.92

### Test Set
- **Accuracy**: 0.80
- **Precision, Recall, F1-Score**:
    - Positive: 0.72, 0.51, 0.60
    - Neutral: 0.78, 0.85, 0.81
    - Negative: 0.83, 0.87, 0.85

## Confusion Matrix (Test Set)
