import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import streamlit as st
import joblib
import os
import PyPDF2
import docx
import pandas as pd
import datetime
import altair as alt

## had to redefine these classes for them to be able to work on website.
class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.apply(lambda t: re.sub(r'[^a-z\s]','', re.sub(r'http\S+','', str(t).lower())).strip())

class TextStats(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        stats=[]
        for doc in X:
            words=doc.split()
            stats.append([len(words), np.mean([len(w) for w in words]) if words else 0, doc.count('.')/max(len(words),1)])
        return np.array(stats)

st.set_page_config(page_title="AI vs Human Text Detection", layout="centered")
st.title("AI vs Human Text Detector")
## for different types of input.
def extract_text_from_pdf(file):
    pdf=PyPDF2.PdfReader(file)
    return " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])

def extract_text_from_docx(file):
    doc=docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

## text cleaner to allow models to process.
def clean_text(text):
    t=str(text).lower()
    t=re.sub(r"http\S+","", t)
    t=re.sub(r"[^a-z\s]","", t)
    t=re.sub(r"\s+"," ", t).strip()
    return t

@st.cache_data
def load_vectorizer(): ##load standalone tf-idf vectorizer for feature display.
    return joblib.load(os.path.join("models","tfidf_vectorizer.pkl"))

@st.cache_data
def load_pipeline(name): ## loads one of the full pipelines.
    return joblib.load(os.path.join("models",name))

vectorizer=load_vectorizer()
pipelines={"SVM":"svm_pipeline.pkl","Decision Tree":"decision_tree_pipeline.pkl","AdaBoost":"adaboost_pipeline.pkl"}
selected_model=st.selectbox("Select a model:", list(pipelines.keys()))
pipeline=load_pipeline(pipelines[selected_model]) ## loads selected pipeline

uploaded_file=st.file_uploader("Upload PDF or Word document", type=["pdf","docx"])
text_input=st.text_area("Or paste your text here:")

if uploaded_file:
    if uploaded_file.name.endswith(".pdf"):
        raw_text=extract_text_from_pdf(uploaded_file)
    else:
        raw_text=extract_text_from_docx(uploaded_file)
else:
    raw_text=text_input

if raw_text and st.button("Classify"):
    cleaned=clean_text(raw_text) #clean text to prepare for pipeline
    input_series=pd.Series([cleaned])
    pred=pipeline.predict(input_series)[0] #prediciton
    conf=pipeline.predict_proba(input_series)[0].max() #condifence score

    st.subheader("Prediction")
    st.markdown(f"Predicted label: `{pred}`")
    st.markdown(f"Confidence score: `{conf:.2f}`")

    #showing top tf-idf features
    vec=vectorizer.transform([cleaned])
    feature_names=vectorizer.get_feature_names_out()
    nz=vec.nonzero()[1]
    top=sorted([(feature_names[i], vec[0,i]) for i in nz], key=lambda x: x[1], reverse=True)[:10]
    st.subheader("TF-IDF Features Detected")
    for w,s in top:
        st.write(f"- {w}: {s:.4f}")
    # if decision tree, show feature importances.
    clf=pipeline.named_steps.get("clf")
    if hasattr(clf,"feature_importances_"):
        imp=clf.feature_importances_
        tfidf_len=len(feature_names)
        tfidf_imp=imp[:tfidf_len]
        idx=np.argsort(tfidf_imp)[-10:][::-1]
        st.subheader("Top Contributing TF-IDF Words")
        for i in idx:
            st.write(f"- {feature_names[i]}: {tfidf_imp[i]:.4f}")
    # report generation.
    ts=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report=f"""Prediction Report

Model: {selected_model}
Prediction: {pred}
Confidence: {conf:.2f}

Top TF-IDF Words:
""" + "\n".join([f"{w}: {s:.4f}" for w,s in top]) + f"""

Input Text (first 1000 chars):
{raw_text[:1000]}

Generated on: {ts}
"""
    fname=f"prediction_report_{ts}.txt"
    with open(fname,"w") as f:
        f.write(report)
    with open(fname,"rb") as f:
        st.download_button("Download report",f,file_name=fname)


# below is the code to display model comparison information and roc curves.
st.divider()
st.subheader("Model Comparison")
cmp_df=pd.read_csv(os.path.join("models","model_comparison.csv"))
st.dataframe(cmp_df)

st.subheader("ROC Curve Comparison")
roc_df=pd.read_csv(os.path.join("models","roc_data.csv"))
chart=(alt.Chart(roc_df).mark_line().encode(x=alt.X("FPR:Q",title="False Positive Rate"),y=alt.Y("TPR:Q",title="True Positive Rate"),color=alt.Color("Model:N")).properties(width=600,height=400))
st.altair_chart(chart,use_container_width=True)
