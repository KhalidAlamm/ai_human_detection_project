# AI vs Human Text Detection Web Application

This repository contains a Streamlit-based web application and supporting machine learning notebook for detecting whether input text was written by AI or a human. It uses three classifiers (SVM, Decision Tree, AdaBoost) trained on TF-IDF and simple linguistic features, tuned via grid search, and packaged as pipelines for real-time inference.

## Repository Structure

ai_human_detection_project/
├── data/
│ ├── AI_vs_huam_train_dataset.xlsx  # Training dataset
│ └── Final_test_data.csv
├── models/
│ ├── adaboost_model.pkl
│ ├── adaboost_pipeline.pkl
│ ├── decision_tree_model.pkl
│ ├── decision_tree_pipeline.pkl
│ ├── model_comparison.csv ## Comparison table of model metrics
│ ├── roc_data.csv # ROC curve data points
│ ├── svm_model.pkl
│ ├── svm_pipeline.pkl
│ └── tfidf_vectorizer.pkl # Fitted TF-IDF vectorizer
├── notebooks/
│ ├── AIvsHumanTextDetection.ipynb #Jupyter notebook
├── app.py # StreamLit application
├── README.md
└── requirements.txt #package dependencies.

## to clone repo


