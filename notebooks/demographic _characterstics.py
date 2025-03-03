# -*- coding: utf-8 -*-
"""
Demographic Characteristics Calculation for MIMIC-III and MIMIC-IV
Author: Nidhi Soley
Date: March 3 2025

This script calculates demographic characteristics (e.g., admission type, location, insurance,
language, marital status, race, and age bucket) for two groups:
    - Normal deliveries (complication = 0)
    - Complicated deliveries (complication = 1)

The script is designed to work for both MIMIC-III and MIMIC-IV datasets. For each dataset, 
an admissions file (demo) is merged with the corresponding notes data. Then age is computed and 
bucketed before generating a table with counts and percentages.
"""

import pandas as pd
import numpy as np
import re
import datetime

# ---------------------------
# Function Definitions
# ---------------------------
def calculate_age(admittime, anchor_year, anchor_age):
    """
    Calculate age as a float value using admittime, anchor_year, and anchor_age.
    The calculation adjusts the days difference by adding the anchor_age in days,
    then converts back to years.
    """
    # Convert anchor_year to a datetime (using January 1st of that year)
    anchor_date = pd.to_datetime(anchor_year.astype(str) + '-01-01')
    days_difference = (admittime - anchor_date).dt.days
    age = (days_difference + anchor_age * 365.25) / 365.25
    return age

def assign_age_bucket(age):
    if pd.isnull(age):
        return "Unknown"
    elif age <= 20:
        return "18-20"
    elif 21 <= age <= 39:
        return "21-39"
    elif 40 <= age <= 50:
        return "40-50"
    elif age >= 51:
        return "51+"
    else:
        return "Unknown"

def generate_population_characteristics(df, id_col='subject_id'):
    """
    Generate a table of population characteristics based on unique subjects.
    For each variable (all columns except id_col), the function computes counts and percentages.
    Returns a consolidated DataFrame.
    """
    result_list = []
    unique_df = df.drop_duplicates(subset=[id_col])
    for col in unique_df.columns:
        if col == id_col:
            continue
        value_counts = unique_df[col].value_counts(dropna=False)
        percentages = (value_counts / len(unique_df)) * 100
        df_temp = pd.DataFrame({
            'Variable': col,
            'Category': value_counts.index,
            'Count': value_counts.values,
            'Percentage': percentages.round(2)
        })
        result_list.append(df_temp)
    characteristics_table = pd.concat(result_list, axis=0)
    characteristics_table = characteristics_table[['Variable', 'Category', 'Count', 'Percentage']]
    return characteristics_table

def merge_demo_notes(demo_df, notes_df, subject_ids):
    """
    Merge an admissions (demo) DataFrame with note data based on a given list of subject_ids.
    Converts the 'admittime' to datetime, calculates age using anchor_year and anchor_age, 
    and assigns an age bucket.
    Returns the merged DataFrame.
    """
    demo_filtered = demo_df[demo_df['subject_id'].isin(subject_ids)]
    merged = pd.merge(demo_filtered, notes_df, on='subject_id', how='inner')
    merged['admittime'] = pd.to_datetime(merged['admittime'])
    merged['age'] = calculate_age(merged['admittime'], merged['anchor_year'], merged['anchor_age'])
    merged['Age_Bucket'] = merged['age'].apply(assign_age_bucket)
    return merged

def generate_demo_statistics(demo_file, notes_normal, notes_comp):
    """
    For a given admissions file (demo_file) and two notes DataFrames (normal and complicated),
    this function merges the demo with each notes DataFrame (using subject_id),
    computes age and age bucket, and returns two population characteristics tables.
    """
    # Load admissions (demo) file
    demo = pd.read_csv(demo_file)
    
    # Get unique subject IDs for each group
    subjects_normal = notes_normal['subject_id'].unique()
    subjects_comp   = notes_comp['subject_id'].unique()
    
    # Merge and process for normal and complicated groups
    df_normal = merge_demo_notes(demo, notes_normal, subjects_normal)
    df_comp   = merge_demo_notes(demo, notes_comp, subjects_comp)
    
    # Select columns of interest
    cols = ['subject_id', 'admission_type', 'admission_location', 'insurance', 'language', 'marital_status', 'race', 'Age_Bucket']
    demo_normal = df_normal[cols]
    demo_comp   = df_comp[cols]
    
    # Generate population characteristics tables based on unique subject_id
    table_normal = generate_population_characteristics(demo_normal)
    table_comp   = generate_population_characteristics(demo_comp)
    return table_normal, table_comp

# ---------------------------
# MAIN CODE: Generate Demographics for MIMIC-IV and MIMIC-III
# ---------------------------

# For MIMIC-IV:
base_path = {your base path}
mimic_iv_demo_file = os.path.join(base_path,'admissions_MMC4.csv')
#  female_notes_normal_with_sh and female_notes_not_normal_with_sh (MIMIC-IV).
pop_char_table_iv_normal, pop_char_table_iv_comp = generate_demo_statistics(mimic_iv_demo_file, 
                                                                           female_notes_normal_with_sh, 
                                                                           female_notes_not_normal_with_sh)

print("MIMIC-IV Normal Deliveries Demographics:")
display(pop_char_table_iv_normal)
print("\nMIMIC-IV Complicated Deliveries Demographics:")
display(pop_char_table_iv_comp)

# For MIMIC-III:
mimic_iii_demo_file = os.path.join(base_path,'admissions_MIMICIII.csv')
# And assume DataFrames for MIMIC-III (e.g., female_notes_normal_with_sh_m3 and female_notes_not_normal_with_sh_m3).
#
pop_char_table_iii_normal, pop_char_table_iii_comp = generate_demo_statistics(mimic_iii_demo_file, 
                                                                             female_notes_normal_with_sh_m3, 
                                                                             female_notes_not_normal_with_sh_m3)

print("MIMIC-III Normal Deliveries Demographics:")
display(pop_char_table_iii_normal)
print("\nMIMIC-III Complicated Deliveries Demographics:")
display(pop_char_table_iii_comp)

