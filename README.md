# 20 Newsgroups Text Classification using TF-IDF and Naive Bayes

## Short Description

This project demonstrates text classification on a subset of the 20 Newsgroups dataset. It uses a pipeline combining `TfidfVectorizer` for feature extraction and `MultinomialNB` for classification to categorize documents into topics like religion, space, and graphics.

## Project Overview

The notebook `day7/model_for_predict_sub.ipynb` performs the following:

1.  **Load Data:** Fetches a specific subset (train and test) of the 20 Newsgroups dataset using `sklearn.datasets.fetch_20newsgroups`, focusing on four categories.
2.  **Model Pipeline:** Creates an `sklearn.pipeline.Pipeline` that includes:
    *   `TfidfVectorizer`: To convert text documents into a matrix of TF-IDF features.
    *   `MultinomialNB`: The Naive Bayes classifier suitable for text data.
3.  **Model Training:** Trains the entire pipeline on the training data.
4.  **Evaluation:**
    *   Predicts categories for the test documents.
    *   Calculates the accuracy score.
    *   Generates and plots a confusion matrix with category names.
5.  **Prediction Function:** Defines a function `predict_category` to classify new, unseen text strings.

## Dataset

*   **20 Newsgroups:** A standard dataset for text classification experiments, collected from Usenet newsgroup posts. This project uses a subset of 4 categories. (Data is fetched automatically by scikit-learn).

## Model Used

*   **Pipeline:** Combines TF-IDF Vectorization and Multinomial Naive Bayes classification.

## Requirements

*   Python 3
*   Scikit-learn
*   Matplotlib
*   Seaborn

## How to Run

1.  Ensure you have the required libraries installed (`pip install scikit-learn matplotlib seaborn`).
2.  Run the Jupyter Notebook `day7/model_for_predict_sub.ipynb`. The dataset will be downloaded automatically if not present.
