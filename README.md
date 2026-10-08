# Automated Visual Quality Inspection System

**DEPI — Digital Egypt Pioneers Initiative**
**Track:** Microsoft ML Engineer

An image-classification system that decides whether a manufactured casting product is **acceptable** or **defective** from a single photo, replacing slow and inconsistent manual inspection.

---

## Table of Contents

1. [Overview](#overview)
2. [Problem Statement](#problem-statement)
3. [Objectives](#objectives)
4. [Dataset](#dataset)
5. [Approach](#approach)
6. [Project Structure](#project-structure)
7. [Tools and Technologies](#tools-and-technologies)
8. [System Requirements](#system-requirements)
9. [Installation](#installation)
10. [Usage](#usage)
11. [Results](#results)
12. [Deliverables](#deliverables)
13. [Optional Extension](#optional-extension)
14. [Team](#team)
15. [Acknowledgements](#acknowledgements)

---

## Overview

This project builds and compares two models on top-view grayscale images of casting impellers:

- a **baseline CNN** built from scratch, and
- a **transfer-learning model** based on a pretrained CNN (ResNet, VGG or another syllabus-covered model).

The output is a documented preprocessing pipeline, trained models, evaluation metrics, and an analysis of the images the model classifies incorrectly.

## Problem Statement

A manufacturing organization needs to identify defective products automatically from images instead of relying entirely on manual inspection.

In casting, defects such as blow holes, pinholes, burrs and shrinkage defects are checked by hand. Manual inspection is time-consuming, is not 100% accurate, and a missed defect can lead to the rejection of a whole order, which causes financial loss.

## Objectives

1. Build an image preprocessing pipeline and justify each choice with an experiment (grayscale vs RGB, resizing, noise filtering, histogram analysis).
2. Build a baseline CNN as the reference point.
3. Apply transfer learning and compare the two training approaches.
4. Reduce overfitting with dropout, L2 regularization, early stopping and model checkpointing.
5. Evaluate classification performance and analyze misclassified images.
6. Deliver a technical report documenting the pipeline, results and findings.

## Dataset

**Real-life industrial dataset of casting product** (Kaggle):
<https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product/data>

Top-view images of a submersible pump impeller, captured under stable lighting. Two classes: `ok_front` (acceptable) and `def_front` (defective).

| Version | Image size | Images | Augmentation |
| --- | --- | --- | --- |
| Main | 300×300, grayscale | 7,348 (train 6,633: 3,758 defective + 2,875 ok; test 715: 453 defective + 262 ok) | Already applied |
| Original | 512×512, grayscale | 1,300 (781 defective + 519 ok) | None |

The main version is already split into `train/` and `test/`, each with `def_front/` and `ok_front/` subfolders.

> **Data leakage note.** Augmentation was applied to the main version before the split, so similar images may appear in both train and test. We plan to split the 1,300 original images ourselves (stratified train/validation/test), augment the training set only, and also report results on the official test folder.

The dataset is **not committed** to this repository. Download it from the link above and place it as described in [Installation](#installation). Check the dataset license and terms of use before use.

## Approach

1. **Exploration and preprocessing:** class counts, sample images, histograms, grayscale vs RGB, resizing, noise filtering.
2. **Baseline CNN** trained from scratch.
3. **Transfer learning** with a pretrained CNN.
4. **Regularization and training control:** dropout, L2, early stopping, model checkpointing.
5. **Comparison** of both models on the same split (accuracy, training time, stability).
6. **Evaluation and error analysis:** precision, recall, F1, confusion matrix, sample predictions, review of misclassified images. Recall on the defective class is prioritized, since missing a defective product costs more than rejecting a good one.

## Project Structure

Planned layout. Folders are created as each stage is completed.

```
visual-quality-inspection/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                      # NOT committed (see Installation)
│   ├── raw/                   # downloaded Kaggle data
│   └── processed/             # our train / val / test split
├── notebooks/
│   ├── 01_eda.ipynb                     # counts, samples, histograms
│   ├── 02_preprocessing.ipynb           # grayscale/RGB, resize, noise filtering
│   ├── 03_baseline_cnn.ipynb
│   ├── 04_transfer_learning.ipynb
│   └── 05_evaluation_error_analysis.ipynb
├── src/
│   ├── data/                  # loading, splitting, augmentation
│   ├── preprocessing/         # preprocessing pipeline
│   ├── models/                # baseline CNN, transfer-learning model
│   ├── train.py               # training with early stopping + checkpointing
│   ├── evaluate.py            # metrics, confusion matrix, predictions
│   └── utils/
├── models/                    # saved checkpoints / final models
├── results/
│   ├── figures/               # training curves, confusion matrices, samples
│   └── metrics/               # metric tables
├── tests/                     # automated tests (if applicable)
└── docs/                      # DEPI documentation, one folder per stage
    ├── 01_planning/           # proposal, plan, roles, risks, KPIs
    ├── 02_literature_review/
    ├── 03_requirements/
    ├── 04_analysis_design/
    ├── 05_implementation/
    ├── 06_testing/            # test plan, test cases, bug reports
    └── 07_final/              # user manual, technical docs, presentation, infographic
```

## Tools and Technologies

Proposed stack, to be confirmed by the team:

| Purpose | Tool |
| --- | --- |
| Language | Python |
| Deep learning | TensorFlow/Keras or PyTorch (one to be chosen) |
| Pretrained models | ResNet or VGG |
| Image processing | OpenCV |
| Metrics and splitting | scikit-learn |
| Training environment | Google Colab or Kaggle notebooks |
| Version control | Git and GitHub |

## System Requirements

- Python 3.10 or later
- A GPU is recommended for training (Google Colab or Kaggle provide one); the code should also run on CPU, more slowly
- Roughly a few GB of free disk space for the dataset and model checkpoints

## Installation

```bash
# 1. Clone the repository
git clone <REPOSITORY_URL>
cd visual-quality-inspection

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

**Get the data:** download the dataset from the Kaggle link in [Dataset](#dataset) and extract it into `data/raw/`, so the class folders are available as `data/raw/.../def_front` and `data/raw/.../ok_front`. Update the paths in the configuration once the final layout is set.

## Usage

> To be completed as the code is written. Planned flow:

```bash
# Explore the data and build the processed split
jupyter notebook notebooks/01_eda.ipynb

# Train the baseline CNN and the transfer-learning model
python src/train.py --model baseline
python src/train.py --model transfer

# Evaluate a trained model
python src/evaluate.py --model transfer
```

## Results

> To be filled after training and evaluation: metrics table, confusion matrix, training curves, sample predictions and error analysis.

## Deliverables

- [ ] Image preprocessing pipeline
- [ ] Baseline CNN model
- [ ] Transfer-learning model
- [ ] Training history
- [ ] Evaluation metrics
- [ ] Confusion matrix
- [ ] Sample predictions
- [ ] Error analysis
- [ ] Technical report

## Optional Extension

Use Azure Machine Learning to train, register and deploy the final model, if time allows.

## Team

| # | Name |
| --- | --- |
| 1 | يوسف سمير عبدالحي الدبيكي |
| 2 | أحمد وحيد ادفاوي أبوزيد |
| 3 | يوسف حمدي عبداللطيف مصطفى |
| 4 | روزان علي محمد عقل |
| 5 | الحسين حسن هنداوي إبراهيم |
| 6 | ملك محمد عبدالنبي محمد |

Roles and responsibilities are documented in `docs/01_planning/`.

## Acknowledgements

- Dataset: *Real-life industrial dataset of casting product* on Kaggle, collected with the help of PILOT TECHNOCAST, Shapar, Rajkot.
- Digital Egypt Pioneers Initiative (DEPI).
