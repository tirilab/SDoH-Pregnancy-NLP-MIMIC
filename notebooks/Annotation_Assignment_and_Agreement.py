# -*- coding: utf-8 -*-
"""
Annotation Assignment and Agreement Analysis for MIMIC-IV Notes
Author: Nidhi Soley
Date: March 3 2025

This script performs two main tasks:
1. It randomly shuffles and assigns each note from the validation set (notes_for_annotation)
   to two out of three annotators, ensuring a balanced distribution.
2. It loads the annotation results from three annotators, performs pairwise merges,
   calculates Cohen’s Kappa for agreement on key labels (Social Support, Occupation, Substance Use),
   and then consolidates the ground truth annotations into a single CSV file.
"""

######################################
# PART 1: ASSIGN NOTES TO ANNOTATORS (Each note gets two annotations)
######################################
import pandas as pd
import numpy as np
np.random.seed(42)

# Assume 'notes_for_annotation' is already defined from previous processing.
# Shuffle the notes to randomize assignment
notes_for_annotation=pd.read_csv(os.path.join(base_path,notes_for_annotation.csv))
notes_for_annotation = notes_for_annotation.sample(frac=1, random_state=42).reset_index(drop=True)

# Define annotators.
annotators = ['Annotator_1', 'Annotator_2', 'Annotator_3']

# Initialize a dictionary to hold lists of assigned notes for each annotator.
assignments = {annotator: [] for annotator in annotators}

# Initialize a counter to track the number of notes assigned per annotator.
annotator_counts = {annotator: 0 for annotator in annotators}

def assign_to_two_annotators(note, counts):
    """
    Assign a given note to the two annotators with the fewest assignments so far.
    Update the counts and return the list of assigned annotator names.
    """
    # Sort annotators by current count (ascending).
    sorted_annotators = sorted(counts, key=counts.get)
    # Select the two with the fewest assignments.
    assigned = sorted_annotators[:2]
    # Update the counts for each assigned annotator.
    for annotator in assigned:
        counts[annotator] += 1
    return assigned

# Loop through each note and assign it to two annotators.
for idx, row in notes_for_annotation.iterrows():
    assigned = assign_to_two_annotators(row, annotator_counts)
    for annotator in assigned:
        assignments[annotator].append(row)

# Convert the assignment lists to DataFrames.
annotator_1_notes = pd.DataFrame(assignments['Annotator_1'])
annotator_2_notes = pd.DataFrame(assignments['Annotator_2'])
annotator_3_notes = pd.DataFrame(assignments['Annotator_3'])

# Print the number of notes assigned to each annotator.
print(f"Annotator 1 will annotate {len(annotator_1_notes)} notes.")
print(f"Annotator 2 will annotate {len(annotator_2_notes)} notes.")
print(f"Annotator 3 will annotate {len(annotator_3_notes)} notes.")

# Save the assigned notes for each annotator.
annotator_1_notes.to_csv(os.path.join(base_path,'annotator_1_notes.csv', index=False))
annotator_2_notes.to_csv(os.path.join(base_path,'annotator_2_notes.csv', index=False))
annotator_3_notes.to_csv(os.path.join(base_path,'annotator_3_notes.csv', index=False))


######################################
# PART 2: ANNOTATOR AGREEMENT ANALYSIS
######################################
from sklearn.metrics import cohen_kappa_score

# Load the annotation CSV files from the three annotators.
ann_JS = pd.read_csv(os.path.join(base_path,'annotation_notes(annotator_1_MB).csv', encoding='latin-1')
ann_MB = pd.read_csv(os.path.join(base_path,'annotation_notes(annotator_2_JS).csv', encoding='latin-1')
ann_NS = pd.read_csv(os.path.join(base_path,'annotation_notes(annotator_3_NS).csv', encoding='latin-1')

# Merge pairwise using 'note_id' to align annotations.
merge_12 = ann_JS.merge(ann_MB, on='note_id', suffixes=('_ann1', '_ann2'))
merge_13 = ann_JS.merge(ann_NS, on='note_id', suffixes=('_ann1', '_ann3'))
merge_23 = ann_MB.merge(ann_NS, on='note_id', suffixes=('_ann2', '_ann3'))

# Print the number of notes annotated by each annotator pair.
print(f"Number of notes annotated by Annotator JS & MB: {len(merge_12)}")
print(f"Number of notes annotated by Annotator JS & NS: {len(merge_13)}")
print(f"Number of notes annotated by Annotator MB & NS: {len(merge_23)}")

# Define the list of labels for agreement analysis.
columns_of_interest = ['Social Support', 'Occupation', 'Substance Use']

def calculate_pairwise_kappa(merged_df, suffix1, suffix2, labels):
    """
    Calculate and print Cohen's Kappa for each label between two annotators.
    """
    for label in labels:
        score = cohen_kappa_score(
            merged_df[f"{label}_{suffix1}"],
            merged_df[f"{label}_{suffix2}"]
        )
        print(f"Cohen's Kappa for {label} between {suffix1} and {suffix2}: {score:.3f}")

print("\nAnnotator JS & MB:")
calculate_pairwise_kappa(merge_12, 'ann1', 'ann2', columns_of_interest)

print("\nAnnotator JS & NS:")
calculate_pairwise_kappa(merge_13, 'ann1', 'ann3', columns_of_interest)

print("\nAnnotator MB & NS:")
calculate_pairwise_kappa(merge_23, 'ann2', 'ann3', columns_of_interest)

######################################
# PART 3: CONSOLIDATE GROUND TRUTH ANNOTATIONS
######################################
# Load a corrected merged file for the JS & MB pair.
merge_12_corr = pd.read_csv('/content/drive/My Drive/Colab Notebooks/nlp project/merge_js_mb_corrected.csv')

# Select only the relevant columns from each merged file.
merge_12_corr = merge_12_corr[['note_id', 'subject_id_ann2', 'anchor_age_ann2', 'text_ann2', 
                               'complication_ann2', 'Social Support_ann2', 'Occupation_ann2', 'Substance Use_ann2']]

merge_13 = merge_13[['note_id', 'subject_id_ann3', 'anchor_age_ann3', 'text_ann3', 
                     'complication_ann3', 'Social Support_ann3', 'Occupation_ann3', 'Substance Use_ann3']]

merge_23 = merge_23[['note_id', 'subject_id_ann3', 'anchor_age_ann3', 'text_ann3', 
                     'complication_ann3', 'Social Support_ann3', 'Occupation_ann3', 'Substance Use_ann3']]

# Rename columns in all merged DataFrames for consistency.
merge_12_corr = merge_12_corr.rename(columns={
    'subject_id_ann2': 'subject_id',
    'anchor_age_ann2': 'anchor_age',
    'text_ann2': 'text',
    'complication_ann2': 'complication',
    'Social Support_ann2': 'social_support',
    'Occupation_ann2': 'occupation',
    'Substance Use_ann2': 'substance_use'
})

merge_13 = merge_13.rename(columns={
    'subject_id_ann3': 'subject_id',
    'anchor_age_ann3': 'anchor_age',
    'text_ann3': 'text',
    'complication_ann3': 'complication',
    'Social Support_ann3': 'social_support',
    'Occupation_ann3': 'occupation',
    'Substance Use_ann3': 'substance_use'
})

merge_23 = merge_23.rename(columns={
    'subject_id_ann3': 'subject_id',
    'anchor_age_ann3': 'anchor_age',
    'text_ann3': 'text',
    'complication_ann3': 'complication',
    'Social Support_ann3': 'social_support',
    'Occupation_ann3': 'occupation',
    'Substance Use_ann3': 'substance_use'
})

# Concatenate the merged DataFrames to create a consolidated annotated dataset.
annotated_notes = pd.concat([merge_12_corr, merge_13, merge_23])

# Reset the index.
annotated_notes = annotated_notes.reset_index(drop=True)

# Save the final consolidated annotated notes.
annotated_notes.to_csv('/content/drive/My Drive/Colab Notebooks/nlp project/annotatated_notes_mimiciv.csv', index=False)

