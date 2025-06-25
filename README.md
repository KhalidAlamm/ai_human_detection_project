# AI vs Human Text Detection Web Application

This repository contains a Streamlit-based web application and supporting machine learning notebook for detecting whether input text was written by AI or a human. It uses three classifiers (SVM, Decision Tree, AdaBoost) trained on TF-IDF and simple linguistic features, tuned via grid search, and packaged as pipelines for real-time inference.

## Repository Structure

ai_human_detection_project/
├── data/
│ ├── AI_vs_huam_train_dataset.xlsx
│ └── Final_test_data.csv
├── models/
│ ├── adaboost_model.pkl
│ ├── adaboost_pipeline.pkl
│ ├── decision_tree_model.pkl
│ ├── decision_tree_pipeline.pkl
│ ├── model_comparison.csv
│ ├── roc_data.csv
│ ├── svm_model.pkl
│ ├── svm_pipeline.pkl
│ └── tfidf_vectorizer.pkl
├── notebooks/
│ ├── AIvsHumanTextDetection.ipynb
│ 
├── app.py
├── requirements.txt
└── README.md

## to clone repo
1. Clone the repository  
   ```bash
   git clone https://github.com/KhalidAlamm/ai_human_detection_project.git
   cd ai_human_detection_project
2. Install dependencies:
    pip install -r requirements.txt

## Running webpage:
    python -m streamlit run app.py

## To re-train, run cells of notebook AIVSHUMANDETECTION.ipynb in order

Dependencies
Python 3.10+

streamlit

scikit-learn

pandas

numpy

scipy

joblib

PyPDF2

python-docx

altair

matplotlib

seaborn

instal via requirements.txt