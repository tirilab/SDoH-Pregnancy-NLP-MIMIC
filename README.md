
# SDoH-Pregnancy-NLP

**SDoH-Pregnancy-NLP** is a repository that leverages natural language processing (NLP) to extract Social Determinants of Health (SDoH) from clinical notes in the MIMIC-III and MIMIC-IV databases. The study focuses on identifying key SDoH factors—Social Support, Occupation, and Substance Use—and examines their association with adverse pregnancy outcomes. Additionally, the repository includes manual annotation, inter-annotator agreement analysis, and demographic characterization of patient populations.

## Overview

Extracting SDoH from unstructured clinical notes is essential for understanding how social factors influence maternal and infant health outcomes. This project implements multiple methods to extract SDoH features, including:

- **Rule-Based Approaches:** Using keyword processors for Social Support and Occupation.
- **Word2Vec Embeddings:** Coupled with machine learning classifiers for Substance Use.
- **ClinicalBERT Embeddings:** Combined with classifiers (Decision Tree, Random Forest, SVC) for capturing contextual nuances in clinical narratives.

The repository also contains code for:
- **Data Preprocessing:** Merging admissions data with clinical notes, calculating age and age buckets.
- **Annotation Assignment:** Randomly assigning notes to annotators (each note is annotated by two out of three annotators) for manual labeling.
- **Annotator Agreement Analysis:** Computing pairwise Cohen’s Kappa scores to assess annotation consistency.
- **Demographic Analysis:** Generating population characteristics tables for both MIMIC-III and MIMIC-IV datasets.

## Repository Structure

```
SDoH-Pregnancy-NLP/
├── README.md
├── notebooks/
│   ├── MIMICIII_SDoH_training_evaluation.py    # Code for training and evaluating models on MIMIC-III
│   ├── MIMICIV_SDoH_validation.py       # Code for validating models on MIMIC-IV
│   └── Annotation_Assignment_and_Agreement.py    # Code for assigning notes, annotation, and agreement analysis
|   └── demographic_characterstics.py #code for calculating demographic characteristics
└── requirements.txt       # Required Python packages and versions
```

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/SDoH-Pregnancy-NLP.git
   cd SDoH-Pregnancy-NLP
   ```

2. **Install dependencies:**

   We recommend using a virtual environment. You can install the required packages using:

   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` includes packages such as `pandas`, `numpy`, `nltk`, `gensim`, `transformers`, `scikit-learn`, and `flashtext`.

## Data

The project uses data from MIMIC database (MIMIC III and IV), the annotated data can be found on [PhysioNet](https://physionet.org/projects/H5W31CeYx0EOsB05USg4/overview/) [1]:
- **MIMICIII_annotation_PregnancySDOH.csv:** Discharge notes with text, and annotations for substance use, social support and occupation.
- **MIMICIV_annotation_PregnancySDOH.csv:** Discharge notes, admissions, and patient demographics.

**Note:** Access to the MIMIC databases requires completion of the required training and adherence to the data use agreement. [MIMIC Training](https://www.physionet.org/content/mimic-iv-note/view-required-training/2.2/)

## Usage

### Training & Evaluation

- **MIMIC-III Models:**  
  Open the `MIMICIII_SDoH_training_evaluation.ipynb` notebook to review and run the code that preprocesses data, trains multiple models for SDoH extraction, and evaluates model performance.

- **MIMIC-IV Validation:**  
  Open the `MIMICIV_SDoH_validation.ipynb` notebook for validating the best performing models on the MIMIC-IV dataset. The best models used in this project are:  
  - Social Support: ClinicalBERT with Decision Tree  
  - Occupation: Keyword Processing classifier  
  - Substance Use: Word2Vec with Random Forest

### Annotation and Agreement Analysis

- **Assignment & Analysis:**  
  The `Annotation_Assignment_and_Agreement.ipynb` notebook contains code to:
  - Randomly assign each note to two annotators.
  - Merge annotations from different annotators.
  - Calculate inter-annotator agreement using Cohen’s Kappa for Social Support, Occupation, and Substance Use.
  - Consolidate final annotated ground truth files.

### Demographic Analysis

- **Demographics:**  
  The `demographic_characterstics.py` script includes functions to merge admissions data with notes, compute age, assign age buckets, and generate population characteristics tables.
## Contributions

Contributions to improve the models, codebase, or documentation are welcome. Please follow the standard GitHub flow with pull requests and clear commit messages. Contact: [Nidhi Soley](nsoley1@jhu.edu) 

## Refereces
1. Soley, N., Bentil, M., Shah, J., Rouhizadeh, M., & Taylor, C. (2025). Annotated Social Determinants of Health Dataset for Adverse Pregnancy Outcomes (version 1.0.0). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/qk2y-wx30
2. Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345.

## License

Code in this repository is released under the MIT License. See the [LICENSE](./LICENSE) file for details.

## Data Access & Licensing

This repository does not include MIMIC data or any protected content. Access to **MIMIC-III** and **MIMIC-IV** requires completion of required training and adherence to the Data Use Agreement. The use of the **annotated SDoH dataset** is governed by the license and terms posted on PhysioNet.

 
