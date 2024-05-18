# -*- coding: utf-8 -*-
"""PROJECT 2Validation for best performing modeL_nsoley1.ipynb

Automatically generated by Colab.

"""

from google.colab import drive
drive.mount('/content/drive')

!pip install --user --upgrade scikit-learn gensim nltk transformers
import nltk
nltk.download('punkt')
import pandas as pd

import pandas as pd
notes_1=pd.read_csv('/content/drive/My Drive/Colab Notebooks/nlp project/NOTEEVENTS.csv')

patients=pd.read_csv('/content/drive/My Drive/Colab Notebooks/nlp project/PATIENTS.csv')

"""# Inlcusion criteria"""

df_patients = patients[patients['GENDER'] == 'F']
female_notes = pd.merge(df_patients, notes_1, on='SUBJECT_ID',how='inner')
female_notes=female_notes[female_notes['CATEGORY']=='Discharge summary']
female_notes

icd_diag=pd.read_csv('/content/drive/My Drive/Colab Notebooks/nlp project/D_ICD_DIAGNOSES.csv')

pregnancy_related = icd_diag[
    icd_diag['LONG_TITLE'].str.contains('pregnancy', case=False) |
    icd_diag['ICD9_CODE'].str.startswith(tuple(['V22.0', 'V22.1', 'V22.2', 'V23.9', 'V24.2', '630', '631', '632', '633', '634', '635', '636', '637','640', '641', '642', '643', '644', '645', '646', '647', '648', '649']))
]

pregnancy_related_excluding_normal_delivery = pregnancy_related[
    ~pregnancy_related['ICD9_CODE'].str.startswith(('650', '651', '652', '653', '654', '655', '656', '657', '658', '659'))
]

pregnancy_related_normal_delivery = pregnancy_related[
    pregnancy_related['ICD9_CODE'].str.startswith(('650', '651', '652', '653', '654', '655', '656', '657', '658', '659'))
]

import pandas as pd
female_notes_not_normal = pd.merge(female_notes, pregnancy_related_excluding_normal_delivery, left_on='ROW_ID_x', right_on='ROW_ID', how='inner')
female_notes_normal=pd.merge(female_notes, pregnancy_related_normal_delivery, left_on='ROW_ID_x', right_on='ROW_ID', how='inner')

"""# Randomly select 40 notes and 10 notes for manually annotating and ground truth creation

"""

import pandas as pd
notes_not_normal_sample = female_notes_not_normal.sample(n=40, random_state=42)[['TEXT','ROW_ID','SUBJECT_ID']]
notes_not_normal_sample['complication']=1
notes_normal_sample = female_notes_normal.sample(n=10, random_state=42)[['TEXT','ROW_ID','SUBJECT_ID']]
notes_normal_sample['complication']=0

combined_notes = pd.concat([notes_not_normal_sample, notes_normal_sample])
combined_notes

"""# Subsetting notes unannotated - for validation"""

not_annotated_notes = all_notes[~all_notes.ROW_ID.isin(combined_notes.ROW_ID)]

not_annotated_notes=not_annotated_notes[['GENDER', 'DOB','TEXT','ROW_ID','SUBJECT_ID','complication']]
not_annotated_notes=not_annotated_notes.reset_index()

notes_model=pd.read_csv('/content/drive/My Drive/Colab Notebooks/nlp project/notes_model.csv')

not_annotated_notes=pd.read_csv('/content/drive/My Drive/Colab Notebooks/nlp project/not_annotated_notes.csv')

!pip3 install --user nltk flashtext

"""# Substance use- keyword processor"""

kp_substance = KeywordProcessor()
keyword_dict = {
    "substanceuse": [
        "alcohol use", "drink alcohol", "consume alcohol", "alcohol consumption", "drink", "drinks", "drinking",
        "tobacco use", "smoke", "cigarette", "nicotine", "smoke", "smokes", "smoking",'pack','packs'
        "drug use", "substance abuse", "illicit drug use", "use drugs", "drug consumption", "drug habit", "drug addiction",
        "no alcohol use", "no drinking", "no alcohol consumption", "not drink alcohol", "not consume alcohol", "not drinking",
        "no tobacco use", "not smoke", "not smoking", "not cigarette", "no nicotine", "not smoking",
        "no drug use", "not use drugs", "no substance abuse", "not illicit drug use", "no drug consumption", "no drug habit", "no drug addiction"
    ]
}
kp_substance.add_keywords_from_dict(keyword_dict)

class SubstanceClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, keywords):
        self.keywords = keywords
        self.classes_ = ['No Substance Use', 'Substance Use']

    def fit(self, X, y, keywords=None):
        if keywords is not None:
            self.keywords = keywords
        return self
    def predict(self, X):
        output = []
        for document in X:
            extractions = self.keywords.extract_keywords(document)
            negations = [word for word in extractions if word.startswith("no ") or word.startswith("denies")]
            substances = [word for word in extractions if not word.startswith("no ")]

            if negations and not substances:
                output.append(0)  # Negation without substance use keywords -> No Substance Use
            elif substances and not negations:
                output.append(1)  # Substance use keywords without negations -> Substance Use
            else:
                output.append(0)  # Otherwise, assume no substance use
        return np.array(output)

    def predict_proba(self, X):
        return self.predict(X)

    def predict_log_proba(self, X):
        proba = self.predict_proba(X)
        return np.log(proba)


classifier_substance = SubstanceClassifier(kp_substance)

#First let's load data
import pandas as pd
from sklearn.model_selection import train_test_split
df=notes_model.copy()
#we will convert diabetic variables to binary
df.loc[df['substance_use']==-1, 'substance_use']=0
#X is refered to as the input data, y the labeled data
X = df.TEXT_mimic.to_list()
y = df['substance_use'].to_list()
#here we are breaking the data into two sets of approximately 50%, each with
X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y, test_size=0.5, shuffle=True, random_state=123)

"""## Validating on unannotated notes - substance use



"""

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
y_pred_test = classifier_substance.predict(not_annotated_notes['TEXT'])

df_model=not_annotated_notes.copy()
df_model['pred_substance_use']=y_pred_test

"""# Occupation- word2vec"""

!pip install --user --upgrade scikit-learn gensim nltk transformers
import nltk
nltk.download('punkt')

!python -m nltk.downloader stopwords

!python -m nltk.downloader wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.parsing.preprocessing import stem_text, preprocess_string, remove_stopwords, strip_multiple_whitespaces
def preprocess_data(notes):
    processed_notes = []
    for note in notes:
        # filtered_tokens = [token for token in note if token not in stopwords]
        # lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
        p=preprocess_string(note, filters=[lambda x: x.lower(), strip_multiple_whitespaces, remove_stopwords, stem_text])
        processed_notes.append(p)
    return processed_notes
df=notes_model.copy()
df.TEXT_mimic=df.TEXT_mimic.fillna(' ')
processed_notes = preprocess_data(df['TEXT_mimic'])

import numpy as np
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier

model = Word2Vec(sentences=processed_notes, vector_size=100, window=2, min_count=4, workers=4)

def note_to_vec(note):
    vecs = [model.wv[word] for word in note if word in model.wv]
    if len(vecs) > 0:
        return np.mean(vecs, axis=0)
    else:
        return np.zeros(model.vector_size)

note_vecs = np.array([note_to_vec(note) for note in processed_notes])
labels=df['Occupation']
X_train, X_test, y_train, y_test = train_test_split(note_vecs, labels, test_size=0.2, random_state=42)
classifiers =RandomForestClassifier(random_state=421)

classifiers.fit(X_train,y_train)

"""## Validating on unannotated notes - occupation

"""

processed_notes_not_ann=preprocess_data(not_annotated_notes['TEXT'])
note_vecs_not_ann = np.array([note_to_vec(note) for note in processed_notes_not_ann])

y_pred_occ=classifiers.predict(note_vecs_not_ann)

df_model['pred_occupation']=y_pred_occ

"""
# Social Support- Clinical BERT"""

!pip install transformers
from transformers import BertTokenizer, BertModel
import torch
from sklearn.tree import DecisionTreeClassifier
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
labels=df['Social Support']
def get_clinical_bert_embeddings(text, model, tokenizer, max_length=512):
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_length - 2:
        tokens = tokens[:max_length - 2]
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = input_ids + [0] * (max_length - len(input_ids))
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    outputs = model(input_ids)
    last_hidden_states = outputs[0]
    return last_hidden_states.squeeze().detach().mean(dim=0).numpy()

clinical_bert_embeddings = [get_clinical_bert_embeddings(' '.join(note), model, tokenizer) for note in processed_notes]

X_train, X_test, y_train, y_test = train_test_split(clinical_bert_embeddings, labels, test_size=0.2, random_state=42)

classifiers_dt = DecisionTreeClassifier(criterion='entropy', class_weight='balanced', random_state=42)

"""## Validating on unannotated notes - social support

"""

#validation
clinical_bert_embeddings_not_ann = [get_clinical_bert_embeddings(' '.join(note), model, tokenizer) for note in processed_notes_not_ann]
classifiers_dt.fit(X_train,y_train)
y_pred_ss=classifiers_dt.predict(clinical_bert_embeddings_not_ann)
df_model['pred_social_support']=y_pred_ss

"""# Regression analysis for just complication ==1 (subjects with complicated pregnancy)"""

import pandas as pd
import statsmodels.api as sm
df = df_model.copy()
df_complication_1 = df[df['complication'] == 1]
dependent_variable = 'complication'
independent_vars = ['pred_substance_use', 'pred_social_support', 'pred_occupation']

independent_vars_with_constant = sm.add_constant(df_complication_1[independent_vars])
model = sm.OLS(df_complication_1[dependent_variable], independent_vars_with_constant)
results = model.fit()
print(results.summary())

import matplotlib.pyplot as plt
complication_1_df = df[df['complication'] == 1]
social_support_counts = complication_1_df['pred_social_support'].value_counts()
plt.figure(figsize=(10, 6))
social_support_counts.plot(kind='bar', color='skyblue')
plt.title('Distribution of pred_social_support for complication = 1')
plt.xlabel('pred_social_support')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

