# -*- coding: utf-8 -*-
"""
MIMIC-IV SDoH Validation
Author: Nidhi Soley
Date: March 3 2025

This notebook validates the best performing SDoH extraction models on a MIMIC-IV dataset.
Best performing models are:
    - Social Support: ClinicalBERT + Decision Tree
    - Occupation: Keyword Processing classifier
    - Substance Use: Word2Vec + Random Forest

The annotated validation set is loaded from 'annotatated_notes_mimiciv.csv'.
"""

######################################
# 1. SETUP: Imports, Drive Mount, and Package Installation
######################################
from google.colab import drive
drive.mount('/content/drive')

# Install required packages (uncomment if needed)
#!pip install --user --upgrade scikit-learn gensim nltk transformers flashtext

import os
import re
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Machine learning and evaluation
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# For keyword processing
from flashtext import KeywordProcessor
from sklearn.base import BaseEstimator, ClassifierMixin

# For ClinicalBERT
import torch
from transformers import AutoTokenizer, AutoModel

######################################
# 2. DATA LOADING & INCLUSION CRITERIA
######################################
# Define base path (adjust if needed)
base_path = {your path}

# Load discharge notes and NOTEEVENTS from MIMIC-IV (validation set)
discharge_df = pd.read_csv(os.path.join(base_path, 'discharge.csv'))
notes_iv = pd.read_csv(os.path.join(base_path, 'NOTEEVENTS.csv'))

# Exclude subjects already present in NOTEEVENTS (if applicable)
discharge_df = discharge_df[~discharge_df['subject_id'].isin(notes_iv['SUBJECT_ID'])]

# Load ICD diagnoses and filter for pregnancy‐related codes (using your ICD filters)
codes = pd.read_csv(os.path.join(base_path, 'D_ICD_DIAGNOSES.csv'))
# Select codes where either SHORT_TITLE or LONG_TITLE contains "pregnancy"
preg_codes_short = codes[codes['SHORT_TITLE'].str.contains("pregnancy", case=False, na=False)]
preg_codes_long  = codes[codes['LONG_TITLE'].str.contains("pregnancy", case=False, na=False)]
pregnancy_codes = pd.concat([preg_codes_short, preg_codes_long])

# Load patient file and filter to female patients
patients = pd.read_csv(os.path.join(base_path, 'patients_m4.csv'))
df_patients = patients[patients['gender'] == 'F']

# Merge female patients with discharge notes
female_notes = pd.merge(df_patients, discharge_df, on='subject_id', how='inner')

# Merge with ICD diagnoses (using pregnancy-related codes)
icd_diag = pd.read_csv(os.path.join(base_path, 'diagnoses_icd.csv'))
icd_diag = icd_diag.merge(pregnancy_codes, left_on='icd_code', right_on='ICD9_CODE', how='inner')

# Merge female notes with ICD information
female_with_icd = pd.merge(female_notes, icd_diag, on='subject_id', how='inner')

# Separate normal and complicated deliveries based on ICD9 prefixes:
# For normal delivery, ICD9 codes typically start with 650-659.
female_notes_normal = female_with_icd[
    female_with_icd['ICD9_CODE'].str.startswith(
        ('650', '651', '652', '653', '654', '655', '656', '657', '658', '659')
    )
]
# For complications, exclude subjects in the normal group and optionally filter by a provided list.
icd9_complicated = [  # Your list of ICD9 codes for complications
    '63311', '64511', '64693', '64231', '64631', '64663', '64801', '64661',
    '64681', '64201', '64662', '64903', '64803', '64233', '64683', '63310',
    '64921', '64223', '64234', '64291', '64911', '64913', '63380', '64203',
    '64323', '64623', '64914', '64931', '67181', '67183', '64671', '63381',
    '64934', '64684', '64951', '64664', '64293', '64901', '64621', '64923',
    '64953', '64521', '64294', '64614', '64204', '64221', '64941', '64943',
    '64651', '64393', '64673', '64933', '64633', '64944', '64232', '64682',
    '67151', '64624', '64093', '67123', '67103', '62981', '64513', '64213',
    '64214', '64611', '64703', '64224', '67182', '64653', '64904', '67124',
    '64612', '64723', '63390', '64321', '64391', '64083', '64622', '64381',
    '67111', '64804', '64383', '67184', '64711', '67153', '67101', '67152',
    '64211', '63320', '63300', '64721', '67121', '67154', '64202', '64613'
]
female_notes_comp = female_with_icd[ female_with_icd['icd_code'].isin(icd9_complicated) ]

# For SDoH extraction, assume we are focusing on the "text" column.
# (If desired, you can extract a "Social History" section from the note text.)
def extract_social_history(text):
    match = re.search(r'Social History:.*', text, re.DOTALL)
    return match.group(0) if match else text

# Apply extraction and drop notes with empty social history (if needed)
female_notes_normal['text_sh'] = female_notes_normal['text'].apply(extract_social_history)
female_notes_comp['text_sh'] = female_notes_comp['text'].apply(extract_social_history)

# For validation, we combine a subset of normal (complication=0) and complicated (complication=1) notes.
female_notes_normal['complication'] = 0
female_notes_comp['complication'] = 1
notes_validation = pd.concat([female_notes_comp, female_notes_normal]).drop_duplicates(subset=['note_id']).reset_index(drop=True)

# For evaluation, we load your manually annotated validation set (60 notes)
annotated_df = pd.read_csv(os.path.join(base_path, 'annotatated_notes_mimiciv.csv'))
# Expected columns: 'text', 'social_support', 'occupation', 'substance_use'
print("Validation set label distribution:")
print(annotated_df[['social_support', 'occupation', 'substance_use']].apply(pd.value_counts))

######################################
# 3. VALIDATION USING BEST-PERFORMING MODELS
######################################

# ------------------------------
# A. SOCIAL SUPPORT: ClinicalBERT + Decision Tree
# ------------------------------
print("\n### Social Support: ClinicalBERT + Decision Tree ###")

# Define function to get ClinicalBERT embeddings (averaged over tokens)
tokenizer_bert = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model_bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

def get_clinical_bert_embedding(text, model, tokenizer, max_length=512):
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_length - 2:
        tokens = tokens[:max_length - 2]
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids += [0] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
    # Average the last hidden states to get a single embedding vector
    last_hidden_states = outputs[0]
    return last_hidden_states.squeeze().detach().mean(dim=0).numpy()

# Generate embeddings for each note in the validation set (using the full text or extracted SH)
annotated_df['bert_embedding'] = annotated_df['text'].apply(lambda x: get_clinical_bert_embedding(x, model_bert, tokenizer_bert))

# Prepare data for classification
X_bert = np.stack(annotated_df['bert_embedding'].values)
y_social = annotated_df['social_support'].replace(-1, 0).values  # ensure labels are 0 or 1

# Split data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X_bert, y_social, test_size=0.2, random_state=42, stratify=y_social)

# Train the Decision Tree classifier
dt_ss = DecisionTreeClassifier(criterion='entropy', class_weight='balanced', random_state=42)
dt_ss.fit(X_train, y_train)
y_pred = dt_ss.predict(X_test)

# Evaluate
print(classification_report(y_test, y_pred, target_names=["No Social Support", "Social Support"]))
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Social Support", "Social Support"]).plot()

# ------------------------------
# B. OCCUPATION: Keyword Processing Classifier
# ------------------------------
print("\n### Occupation: Keyword Processing Classifier ###")

# Build a KeywordProcessor with occupation-related keywords
kp_occup = KeywordProcessor()
occup_keywords = [
    "employed", "retired", "work as", "worked in", "works in", "works as", "worked", 
    "job", "employment", "quit", "unemployed", "jobless", "working", "works"
]
kp_occup.add_keywords_from_list(occup_keywords)

# Define a simple keyword classifier (same logic as used during training)
class KeywordClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, keyword_processor):
        self.keyword_processor = keyword_processor
        self.classes_ = ['No Occupation', 'Occupation']

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        preds = []
        for doc in X:
            found = self.keyword_processor.extract_keywords(doc)
            preds.append(1 if len(found) > 0 else 0)
        return np.array(preds)

# Instantiate the classifier and apply to validation text
classifier_occup = KeywordClassifier(kp_occup)
X_text = annotated_df['text'].tolist()
y_occ = annotated_df['occupation'].replace(-1, 0).values

# Get predictions and evaluate
y_pred_occ = classifier_occup.predict(X_text)
print(classification_report(y_occ, y_pred_occ, target_names=["No Occupation", "Occupation"]))
cm_occ = confusion_matrix(y_occ, y_pred_occ)
ConfusionMatrixDisplay(confusion_matrix=cm_occ, display_labels=["No Occupation", "Occupation"]).plot()

# ------------------------------
# C. SUBSTANCE USE: Word2Vec + Random Forest
# ------------------------------
print("\n### Substance Use: Word2Vec + Random Forest ###")

# Preprocess text for Word2Vec: tokenize, remove stopwords, and lemmatize.
stopwords_list = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = text.split()  # simple split; replace with more robust tokenization if desired
    tokens = [token.lower() for token in tokens if token.lower() not in stopwords_list]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

# Apply preprocessing to the validation text
annotated_df['tokens'] = annotated_df['text'].apply(preprocess_text)

# Train a Word2Vec model on the validation text (if you have a larger corpus you could use that)
from gensim.models import Word2Vec
w2v_model = Word2Vec(sentences=annotated_df['tokens'], vector_size=150, window=4, min_count=1, workers=6)

def note_to_vector(tokens, model):
    vecs = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size)

# Convert tokens to vectors for each note
annotated_df['w2v_vector'] = annotated_df['tokens'].apply(lambda tokens: note_to_vector(tokens, w2v_model))
X_w2v = np.stack(annotated_df['w2v_vector'].values)
y_sub = annotated_df['substance_use'].replace(-1, 0).values

# Split data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X_w2v, y_sub, test_size=0.2, random_state=42, stratify=y_sub)

# Train a Random Forest classifier
rf_sub = RandomForestClassifier(random_state=42)
rf_sub.fit(X_train, y_train)
y_pred_sub = rf_sub.predict(X_test)

# Evaluate
print(classification_report(y_test, y_pred_sub, target_names=["No Substance Use", "Substance Use"]))
cm_sub = confusion_matrix(y_test, y_pred_sub)
ConfusionMatrixDisplay(confusion_matrix=cm_sub, display_labels=["No Substance Use", "Substance Use"]).plot()


