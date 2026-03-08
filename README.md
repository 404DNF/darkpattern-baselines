# Dark Pattern Classification Experiments

This repository runs three experiments in sequence with 5-fold Stratified Cross-Validation.

## Experiment Structure

1. **Binary Classification**
   - Task: NDP vs DP
   - Target: `label`
   - Models:
     - Classical NLP: Logistic Regression, SVM, Random Forest, LightGBM
     - PLMs: BERT / RoBERTa / ALBERT / XLNet

2. **Predicate Classification**
   - Task: fine-grained multi-class classification
   - Target: `predicate_id`
   - Includes:
     - Accuracy
     - Macro-F1
     - Weighted-F1
     - Derived-type metrics from predicate predictions

3. **Type Classification**
   - Task: coarse-grained multi-class classification
   - Target: `type_id`

---

## Required CSV Columns

The input CSV must contain:

- `String` or `text`
- `label`
- `predicate`
- `type` or `Type`

`String` is automatically renamed to `text`.

---

## Recommended Environment

Tested environment:

- Python `3.10.13`
- torch `2.6.0`
- transformers `4.56.2`
- accelerate `1.13.0`
- datasets `2.21.0`
- scikit-learn `1.5.2`
- lightgbm `4.5.0`
- pandas `2.2.3`
- numpy `2.0.2`
- safetensors `0.4.5`

### macOS only

LightGBM requires OpenMP runtime:

```bash
brew install libomp
```

## install

```bash
git clone https://github.com/your-repo/darkpattern-baselines.git
cd darkpattern-baselines

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
python main.py --csv_path ./data/contextual_total.csv --output_dir ./outputs
```

## Output Structure
1 ) Binary Classification
```bash
outputs/
├── classical_5fold.csv
├── plm_5fold.csv
├── darkpattern_baselines_5fold.csv
└── plm_cv/
    └── plm_results_partial.csv
```
2 ) Predicate + 3) Type Classification
```bash
outputs/hierarchical/
├── hier_plm_results_partial.csv
├── hier_plm_results_final.csv
└── hier_plm_classwise_long.csv
```
In hier_plm_results_final.csv:
	•	task == "predicate" → Predicate classification results
	•	task == "type" → Type classification results

## Notes
All experiments use 5-fold Stratified Cross-Validation.
The script runs in this order:
1. Binary classification
2. Predicate classification
3. Type classification
- Transformer checkpoints are downloaded from Hugging Face on first run.
- Large PLMs may exceed local GPU / MPS memory. In that case, use a GPU environment such as Colab.
