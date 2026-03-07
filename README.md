# Dark Pattern Detection Baselines
This repository provides baseline models for dark pattern detection using 5-fold Stratified Cross-Validation.

Two types of approaches are included:
	•	Classical NLP models (Bag-of-Words features)
	•	Transformer-based pretrained language models (PLMs)

The goal is to provide reproducible baseline results for binary dark pattern classification.

⸻

## Models

### Classical NLP (BoW)

Text is converted to bag-of-words features (1-2 grams) and used with the following classifiers:
	•	Logistic Regression
	•	SVM (RBF kernel)
	•	Random Forest
	•	LightGBM


### Transformer PLMs

Each model is fine-tuned with 5-fold CV using HuggingFace Trainer.

Supported models:
	•	BERT-base / BERT-large
	•	RoBERTa-base / RoBERTa-large
	•	ALBERT-base / ALBERT-large
	•	XLNet-base / XLNet-large

⸻

## Expected Input CSV Format

The CSV file must contain:
	•	String or text → input text
	•	label → binary label (0 / 1)

Example:

String	label
This site is misleading.	1
This is a normal interface.	0

If a column named String exists, it is automatically renamed to text.

⸻

## Recommended Environment

Tested environment:
	•	Python 3.10.13
	•	torch 2.6.0
	•	transformers 4.56.2
	•	accelerate 1.13.0
	•	scikit-learn 1.5.2
	•	lightgbm 4.5.0

Operating systems tested:
	•	macOS
	•	Linux

GPU is optional.
The code automatically falls back to CPU if CUDA is not available.

### macOS Additional Setup

LightGBM requires OpenMP runtime on macOS.

Install it with:

brew install libomp

Without this step LightGBM may fail with:

Library not loaded: libomp.dylib


⸻

## Installation

Clone the repository and create a virtual environment.

git clone https://github.com/your-repo/darkpattern-baselines.git
cd darkpattern-baselines

Create environment:

python -m venv venv
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt


⸻

Run

Place your dataset inside the data/ directory.

Example:

data/contextual_total.csv

Run the experiment:

python main.py --csv_path ./data/contextual_total.csv --output_dir ./outputs


⸻

## Output Files

Results are automatically saved in the outputs/ directory.

Generated files:

outputs/
 ├── classical_5fold.csv
 ├── plm_5fold.csv
 ├── darkpattern_baselines_5fold.csv
 └── plm_cv/

Descriptions:

File	Description
classical_5fold.csv	Classical NLP baseline results
plm_5fold.csv	Transformer model results
darkpattern_baselines_5fold.csv	Combined result table
plm_cv/	Intermediate results during PLM training


⸻

### Cross-Validation Protocol

All experiments use:
	•	5-fold Stratified Cross-Validation
	•	Binary classification
	•	Evaluation metrics:

Metric
Accuracy
Precision
Recall
F1
ROC-AUC

Mean and standard deviation across folds are reported.

⸻

### Notes

First run may download models

Transformer models are downloaded from HuggingFace:

huggingface.co

This happens automatically on the first run.

LightGBM fallback

If LightGBM cannot be loaded due to system dependencies, the script will skip LightGBM and continue running other models.

⸻

Project Structure

darkpattern-baselines/
├── main.py
├── requirements.txt
├── README.md
├── src/
│   ├── classical_models.py
│   ├── plm_models.py
│   ├── metrics.py
│   ├── data_utils.py
│   └── utils.py
├── data/
└── outputs/


⸻

Reproducibility

To reproduce the experiments:
	1.	Install the dependencies
	2.	Prepare the dataset
	3.	Run main.py

All results will be generated automatically.

