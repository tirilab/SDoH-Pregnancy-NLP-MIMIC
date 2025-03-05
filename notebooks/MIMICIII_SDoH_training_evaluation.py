# -*- coding: utf-8 -*-
"""
PROJECT: SDoH Training and Evaluation on MIMIC Data
Author: Nidhi Soley
Date: March 3 2025

This notebook loads the MIMIC-III data, applies inclusion criteria, merges clinical notes with ICD diagnosis info,
and then trains/evaluates three types of models for extracting Social Determinants of Health (SDoH) factors:
  1. Rule-based keyword classifiers (using FlashText)
  2. Word2Vec embedding models
  3. ClinicalBERT embedding models

SDoH factors include Social Support, Occupation, and Substance Use.
"""

######################################
# 1. SETUP: Imports, Drive Mount, and Package Installation
######################################
# Mount Google Drive (if using Colab)
from google.colab import drive
drive.mount('/content/drive')

# Install required packages if needed (uncomment below)
#!pip install --user --upgrade scikit-learn gensim nltk transformers flashtext

import os
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# For keyword processing
from flashtext import KeywordProcessor

# Machine Learning models and evaluation
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# For ClinicalBERT
import torch
from transformers import AutoTokenizer, AutoModel

######################################
# 2. DATA LOADING AND PREPROCESSING
######################################
# Define base path (adjust as needed)
base_path = {your path}

# Load datasets
notes_df = pd.read_csv(os.path.join(base_path, 'NOTEEVENTS.csv'))
patients_df = pd.read_csv(os.path.join(base_path, 'PATIENTS.csv'))
icd_diag = pd.read_csv(os.path.join(base_path, 'D_ICD_DIAGNOSES.csv'))

# -- Inclusion Criteria --
# Filter female patients and keep only discharge summaries
df_patients = patients_df[patients_df['GENDER'] == 'F']
female_notes = pd.merge(df_patients, notes_df, on='SUBJECT_ID', how='inner')
female_notes = female_notes[female_notes['CATEGORY'] == 'Discharge summary']

# Select pregnancy-related ICD codes using string matching and prefix conditions
pregnancy_related = icd_diag[
    icd_diag['LONG_TITLE'].str.contains('pregnancy', case=False) |
    icd_diag['ICD9_CODE'].str.startswith(tuple(['V22.0', 'V22.1', 'V22.2', 'V23.9', 'V24.2', 
                                                 '630', '631', '632', '633', '634', '635', 
                                                 '636', '637','640', '641', '642', '643', 
                                                 '644', '645', '646', '647', '648', '649']))
]

# Split pregnancy diagnoses into those indicating complications vs. normal delivery
pregnancy_excl_normal = pregnancy_related[
    ~pregnancy_related['ICD9_CODE'].str.startswith(('650', '651', '652', '653', '654', 
                                                     '655', '656', '657', '658', '659'))
]
pregnancy_normal = pregnancy_related[
    pregnancy_related['ICD9_CODE'].str.startswith(('650', '651', '652', '653', '654', 
                                                    '655', '656', '657', '658', '659'))
]

# Merge clinical notes with pregnancy diagnosis info using appropriate keys
female_notes_not_normal = pd.merge(female_notes, pregnancy_excl_normal, left_on='ROW_ID_x', right_on='ROW_ID', how='inner')
female_notes_not_normal = female_notes_not_normal.drop_duplicates(subset=['note_id'], keep='last')

female_notes_normal = pd.merge(female_notes, pregnancy_normal, left_on='ROW_ID_x', right_on='ROW_ID', how='inner')
female_notes_normal = female_notes_normal.drop_duplicates(subset=['note_id'], keep='last')

# Create sample data with a "complication" flag (1 = complication, 0 = normal)
notes_not_normal_sample = female_notes_not_normal[['TEXT', 'ROW_ID', 'SUBJECT_ID', 'GENDER', 'DOB']].copy()
notes_not_normal_sample['complication'] = 1
notes_normal_sample = female_notes_normal[['TEXT', 'ROW_ID', 'SUBJECT_ID', 'GENDER', 'DOB']].copy()
notes_normal_sample['complication'] = 0

# Combine and reset index
combined_notes = pd.concat([notes_not_normal_sample, notes_normal_sample]).reset_index(drop=True)

# Load manually annotated notes (if available)
mimic_annotations = pd.read_csv(os.path.join(base_path, 'forannotation_frommimic.csv'))
mimic_annotations = mimic_annotations.join(combined_notes, how='inner', lsuffix='_mimic', rsuffix='_combined')
mimic_annotations = mimic_annotations[['TEXT_mimic', 'ROW_ID_mimic', 'SUBJECT_ID_mimic', 
                                       'Social Support', 'Occupation', 'Tobacco', 
                                       'Alcohol', 'Drug', 'TEXT_combined', 'complication']]

# Load model development data (annotated notes)
notes_model = pd.read_csv(os.path.join(base_path, 'notes_model.csv'))

######################################
# 3. RULE-BASED (KEYWORD) CLASSIFIERS
######################################
# Define a generic KeywordClassifier (for Social Support and Occupation)
class KeywordClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, keyword_processor, class_labels):
        """
        keyword_processor: instance of flashtext.KeywordProcessor with keywords added.
        class_labels: list of two labels (e.g., ['No X', 'X'])
        """
        self.keyword_processor = keyword_processor
        self.classes_ = class_labels

    def fit(self, X, y=None, **kwargs):
        return self

    def predict(self, X):
        preds = []
        for document in X:
            found = self.keyword_processor.extract_keywords(document)
            preds.append(1 if len(found) > 0 else 0)
        return np.array(preds)

    def predict_proba(self, X):
        return self.predict(X)

# Specialized classifier for Substance Use that handles negations
class SubstanceClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, keyword_processor, class_labels=['No Substance Use', 'Substance Use']):
        self.keyword_processor = keyword_processor
        self.classes_ = class_labels

    def fit(self, X, y=None, **kwargs):
        return self

    def predict(self, X):
        preds = []
        for document in X:
            found = self.keyword_processor.extract_keywords(document)
            negations = [kw for kw in found if kw.startswith("no ") or kw.startswith("denies")]
            substances = [kw for kw in found if not kw.startswith("no ")]
            if substances and not negations:
                preds.append(1)
            else:
                preds.append(0)
        return np.array(preds)

    def predict_proba(self, X):
        return self.predict(X)

# --------------------------
# Social Support Keyword Classifier
# --------------------------
kp_social = KeywordProcessor()
social_support_keywords = [
    "live with someone", "live with family", "live with friends", "lives with husband",
    "supportive", "support", "married", "husband", "boyfriend"
]
kp_social.add_keywords_from_list(social_support_keywords)
classifier_ss = KeywordClassifier(kp_social, ['No Social support', 'Social Support'])

# --------------------------
# Occupation Keyword Classifier
# --------------------------
kp_occup = KeywordProcessor()
occupation_keywords = [
    "employed", "retired", "work as", "worked in", "works in", "works as", "worked", 
    "job", "employment", "quit job",'stopped working', "unemployed", "jobless", "working"
]
kp_occup.add_keywords_from_list(occupation_keywords)
classifier_occup = KeywordClassifier(kp_occup, ['No Occupation', 'Occupation'])

# --------------------------
# Substance Use Keyword Classifier
# --------------------------
kp_substance = KeywordProcessor()
substance_keywords = [
    "alcohol use", "drink alcohol", "consume alcohol", "alcohol consumption", "drink", "drinks", "drinking",
    "tobacco use", "smokes", "cigarette smoker", "nicotine", "smokes", "smoking", "pack", "packs", 'tob use','marijuana','CBD',
  'cannabis', 'cannabinoid use', 'marijuana use', 'marihuana', 'tetrahydrocannabinol', 'cannabidiol', 'cannabigerole', 'cannabinol'
    "drug use", "substance abuse", "illicit drug use", "use drugs", "drug consumption", "drug habit", "drug addiction", "cannabis abuse",
  
  
]
kp_substance.add_keywords_from_list(substance_keywords)
classifier_substance = SubstanceClassifier(kp_substance)

# --------------------------
# Evaluation of Keyword Models on Annotated Data (notes_model)
# --------------------------
df_model = notes_model.copy()
df_model.loc[df_model['Social Support'] == -1, 'Social Support'] = 0
df_model.loc[df_model['Occupation'] == -1, 'Occupation'] = 0
df_model.loc[df_model['substance_use'] == -1, 'substance_use'] = 0

X_text = df_model['TEXT_mimic'].tolist()

# Evaluate Social Support
y_ss = df_model['Social Support'].tolist()
X_train_ss, X_test_ss, y_train_ss, y_test_ss = train_test_split(X_text, y_ss, stratify=y_ss, test_size=0.4, random_state=1234)
y_pred_ss = classifier_ss.predict(X_test_ss)
print("Social Support Classification Report (Keyword):")
print(classification_report(y_test_ss, y_pred_ss, target_names=classifier_ss.classes_))
cm_ss = confusion_matrix(y_test_ss, y_pred_ss)
ConfusionMatrixDisplay(confusion_matrix=cm_ss, display_labels=classifier_ss.classes_).plot()

# Evaluate Occupation
y_occ = df_model['Occupation'].tolist()
X_train_occ, X_test_occ, y_train_occ, y_test_occ = train_test_split(X_text, y_occ, stratify=y_occ, test_size=0.5, random_state=1234)
y_pred_occ = classifier_occup.predict(X_test_occ)
print("Occupation Classification Report (Keyword):")
print(classification_report(y_test_occ, y_pred_occ, target_names=classifier_occup.classes_))
cm_occ = confusion_matrix(y_test_occ, y_pred_occ)
ConfusionMatrixDisplay(confusion_matrix=cm_occ, display_labels=classifier_occup.classes_).plot()

# Evaluate Substance Use
y_sub = df_model['substance_use'].tolist()
X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(X_text, y_sub, stratify=y_sub, test_size=0.5, random_state=123)
y_pred_sub = classifier_substance.predict(X_test_sub)
print("Substance Use Classification Report (Keyword):")
print(classification_report(y_test_sub, y_pred_sub, target_names=classifier_substance.classes_))
cm_sub = confusion_matrix(y_test_sub, y_pred_sub)
ConfusionMatrixDisplay(confusion_matrix=cm_sub, display_labels=classifier_substance.classes_).plot()

######################################
# 4. WORD2VEC EMBEDDING MODELS
######################################
# Preprocessing function for Word2Vec
lemmatizer = WordNetLemmatizer()
stopwords_list = set(stopwords.words('english'))

def preprocess_notes(text_series):
    """
    Tokenizes, removes stopwords, and lemmatizes each note.
    Returns a list of token lists.
    """
    processed = []
    for note in text_series:
        tokens = note.split()  # Simple tokenization; consider nltk.word_tokenize for improved results.
        filtered = [token.lower() for token in tokens if token.lower() not in stopwords_list]
        lemmatized = [lemmatizer.lemmatize(token) for token in filtered]
        processed.append(lemmatized)
    return processed

# Preprocess notes from TEXT_combined column for Word2Vec models
df_model_wc = notes_model.copy()
processed_notes = preprocess_notes(df_model_wc['TEXT_combined'])

# Train Word2Vec model
from gensim.models import Word2Vec
w2v_model = Word2Vec(sentences=processed_notes, vector_size=150, window=4, min_count=1, workers=6)

def note_to_vec(note, model):
    """
    Converts a token list into a vector by averaging word vectors.
    """
    vecs = [model.wv[word] for word in note if word in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size)

# Convert all notes into vectors
note_vecs = np.array([note_to_vec(note, w2v_model) for note in processed_notes])

# Define classifiers and scoring metrics
classifiers = [
    RandomForestClassifier(random_state=42),
    SVC(kernel='rbf', class_weight='balanced'),
    DecisionTreeClassifier(criterion='entropy', class_weight='balanced', random_state=42)
]
scoring = {'acc': 'accuracy', 'precision': 'precision', 'recall': 'recall', 'f1': 'f1'}

# --- Word2Vec for Social Support ---
labels_ss_wc = df_model_wc['Social Support']
X_train_wc_ss, X_test_wc_ss, y_train_wc_ss, y_test_wc_ss = train_test_split(note_vecs, labels_ss_wc, test_size=0.2, random_state=42)
print("\nWord2Vec + ML Classifiers for Social Support:")
for clf in classifiers:
    scores = cross_validate(clf, X_train_wc_ss, y_train_wc_ss, scoring=scoring, cv=10, return_train_score=False)
    print("Classifier:", clf.__class__.__name__)
    print("  Accuracy:  {:.3f}".format(scores['test_acc'].mean()))
    print("  Precision: {:.3f}".format(scores['test_precision'].mean()))
    print("  Recall:    {:.3f}".format(scores['test_recall'].mean()))
    print("  F1 Score:  {:.3f}".format(scores['test_f1'].mean()))
    print()

# --- Word2Vec for Occupation ---
labels_occ_wc = df_model_wc['Occupation']
X_train_wc_occ, X_test_wc_occ, y_train_wc_occ, y_test_wc_occ = train_test_split(note_vecs, labels_occ_wc, test_size=0.2, random_state=42)
print("\nWord2Vec + ML Classifiers for Occupation:")
for clf in classifiers:
    scores = cross_validate(clf, X_train_wc_occ, y_train_wc_occ, scoring=scoring, cv=10, return_train_score=False)
    print("Classifier:", clf.__class__.__name__)
    print("  Accuracy:  {:.3f}".format(scores['test_acc'].mean()))
    print("  Precision: {:.3f}".format(scores['test_precision'].mean()))
    print("  Recall:    {:.3f}".format(scores['test_recall'].mean()))
    print("  F1 Score:  {:.3f}".format(scores['test_f1'].mean()))
    print()

# --- Word2Vec for Substance Use ---
labels_sub_wc = df_model_wc['substance_use']
X_train_wc_sub, X_test_wc_sub, y_train_wc_sub, y_test_wc_sub = train_test_split(note_vecs, labels_sub_wc, test_size=0.2, random_state=42)
print("\nWord2Vec + ML Classifiers for Substance Use:")
for clf in classifiers:
    scores = cross_validate(clf, X_train_wc_sub, y_train_wc_sub, scoring=scoring, cv=10, return_train_score=False)
    print("Classifier:", clf.__class__.__name__)
    print("  Accuracy:  {:.3f}".format(scores['test_acc'].mean()))
    print("  Precision: {:.3f}".format(scores['test_precision'].mean()))
    print("  Recall:    {:.3f}".format(scores['test_recall'].mean()))
    print("  F1 Score:  {:.3f}".format(scores['test_f1'].mean()))
    print()

######################################
# 5. CLINICAL BERT EMBEDDING MODELS
######################################
# Load pre-trained ClinicalBERT tokenizer and model
tokenizer_bert = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model_bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

def get_clinical_bert_embedding(text, model, tokenizer, max_length=512):
    """
    Generate a ClinicalBERT embedding for a given text by tokenizing, truncating,
    and averaging the hidden states.
    """
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_length - 2:
        tokens = tokens[:max_length - 2]
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = input_ids + [0] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
    last_hidden_states = outputs[0]
    return last_hidden_states.squeeze().mean(dim=0).numpy()

# Generate ClinicalBERT embeddings for each note (joining tokens back to text)
clinical_bert_embeddings = [get_clinical_bert_embedding(' '.join(note), model_bert, tokenizer_bert)
                            for note in processed_notes]

# --- ClinicalBERT for Social Support ---
labels_ss_bert = df_model_wc['Social Support']
X_train_bert_ss, X_test_bert_ss, y_train_bert_ss, y_test_bert_ss = train_test_split(clinical_bert_embeddings, labels_ss_bert, test_size=0.2, random_state=42)
classifiers_bert_ss = [
    RandomForestClassifier(random_state=42),
    SVC(kernel='linear', class_weight='balanced'),
    DecisionTreeClassifier(criterion='entropy', class_weight='balanced', random_state=42)
]
print("\nClinicalBERT + ML Classifiers for Social Support:")
for clf in classifiers_bert_ss:
    scores = cross_validate(clf, X_train_bert_ss, y_train_bert_ss, scoring=scoring, cv=5, return_train_score=False)
    print("Classifier:", clf.__class__.__name__)
    print("  Accuracy:  {:.3f}".format(scores['test_acc'].mean()))
    print("  Precision: {:.3f}".format(scores['test_precision'].mean()))
    print("  Recall:    {:.3f}".format(scores['test_recall'].mean()))
    print("  F1 Score:  {:.3f}".format(scores['test_f1'].mean()))
    print()

# --- ClinicalBERT for Occupation ---
labels_occ_bert = df_model_wc['Occupation']
X_train_bert_occ, X_test_bert_occ, y_train_bert_occ, y_test_bert_occ = train_test_split(clinical_bert_embeddings, labels_occ_bert, test_size=0.2, random_state=42)
classifiers_bert_occ = [
    RandomForestClassifier(random_state=42),
    SVC(kernel='rbf', class_weight='balanced'),
    DecisionTreeClassifier(criterion='entropy', class_weight='balanced', random_state=142)
]
print("\nClinicalBERT + ML Classifiers for Occupation:")
for clf in classifiers_bert_occ:
    scores = cross_validate(clf, X_train_bert_occ, y_train_bert_occ, scoring=scoring, cv=5, return_train_score=False)
    print("Classifier:", clf.__class__.__name__)
    print("  Accuracy:  {:.3f}".format(scores['test_acc'].mean()))
    print("  Precision: {:.3f}".format(scores['test_precision'].mean()))
    print("  Recall:    {:.3f}".format(scores['test_recall'].mean()))
    print("  F1 Score:  {:.3f}".format(scores['test_f1'].mean()))
    print()

# --- ClinicalBERT for Substance Use ---
labels_sub_bert = df_model_wc['substance_use']
X_train_bert_sub, X_test_bert_sub, y_train_bert_sub, y_test_bert_sub = train_test_split(clinical_bert_embeddings, labels_sub_bert, test_size=0.2, random_state=42)
classifiers_bert_sub = [
    RandomForestClassifier(random_state=42),
    SVC(kernel='poly', class_weight='balanced'),
    DecisionTreeClassifier(criterion='entropy', class_weight='balanced', random_state=42)
]
print("\nClinicalBERT + ML Classifiers for Substance Use:")
for clf in classifiers_bert_sub:
    scores = cross_validate(clf, X_train_bert_sub, y_train_bert_sub, scoring=scoring, cv=5, return_train_score=False)
    print("Classifier:", clf.__class__.__name__)
    print("  Accuracy:  {:.3f}".format(scores['test_acc'].mean()))
    print("  Precision: {:.3f}".format(scores['test_precision'].mean()))
    print("  Recall:    {:.3f}".format(scores['test_recall'].mean()))
    print("  F1 Score:  {:.3f}".format(scores['test_f1'].mean()))
    print()

