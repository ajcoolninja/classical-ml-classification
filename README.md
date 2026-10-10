# Classical ML Classification: UCI Bank Marketing

This project compares classical machine-learning classifiers for predicting whether a bank client subscribed to a term deposit. The workflow covers exploratory analysis, leakage-aware preprocessing, model fitting, training-only cross-validation, and final test-set evaluation.

## Dataset

The project uses the **Bank Marketing** dataset from the UCI Machine Learning Repository:

- Moro, S., Cortez, P., and Rita, P. (2014), *A Data-Driven Approach to Predict the Success of Bank Telemarketing*.
- Source: [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- Local file: [`data/bank/bank-full.csv`](data/bank/bank-full.csv)

The target is `y`, where `yes` means that the client subscribed and `no` means that the client did not. In this data, the target is approximately 88% `no` and 12% `yes`, so accuracy by itself is not an adequate measure of performance.

## Models

The final candidate comparison contains:

- Balanced logistic regression
- Balanced decision tree
- Normal random forest
- Balanced RBF SVM
- Soft-voting ensemble using the tuned component pipelines

The random forest candidate is the normal, not class-balanced, version because that was the selected configuration in the existing workflow. The balanced alternatives use `class_weight="balanced"`.

## Preprocessing

- The data is split into training and test sets with an 80/20 split, `random_state=42`, and stratification.
- Categorical features use `OneHotEncoder(drop="first", handle_unknown="ignore")`.
- Linear and SVM pipelines use `RobustScaler`; tree pipelines leave numeric features unscaled.
- `pdays=-1` is retained as a separate `pdays_not_contacted` indicator, while the sentinel is replaced with the median `pdays` among previously contacted clients.
- Preprocessing is inside scikit-learn pipelines, so transformations are fitted within each training fold during cross-validation.
- `duration` is excluded from the ordinary numeric preprocessing because it is only known after a call and may not be available in a genuine pre-call prediction setting.

## Evaluation

The final cross-validation comparison uses five-fold stratified cross-validation on the training data only. It reports:

- **F1-score**, which balances precision and recall for the positive `yes` class.
- **Average precision**, computed by scikit-learn with `scoring="average_precision"`. This summarizes precision across score thresholds; it should not be casually treated as identical to every possible trapezoidal PR-AUC implementation.

ROC-AUC, precision, recall, F1-score, average precision, confusion matrices, and precision-recall curves were also calculated for the selected models on the test set. ROC-AUC and average precision use continuous positive-class scores rather than hard predictions.

### Final training cross-validation results

These values are from the stored `final_cv_comparison` output and are sorted by mean CV F1:

| Model | Mean CV F1 | Mean CV average precision |
|---|---:|---:|
| Voting ensemble (tuned, balanced components) | 0.454514 | 0.433307 |
| Decision tree (balanced, tuned) | 0.410056 | 0.381897 |
| SVM (balanced, tuned) | 0.407847 | 0.360002 |
| Logistic regression (balanced, tuned) | 0.372850 | 0.398059 |
| Random forest (normal, tuned) | 0.315658 | 0.425630 |

The voting ensemble has the highest mean CV F1 and mean CV average precision in this comparison. The random forest has a lower mean CV F1 but a comparatively high mean CV average precision. These aggregate results do not imply that one model is best at every classification threshold.

### Precision-recall tradeoffs

The stored test evaluation illustrates different operating tradeoffs for the positive `yes` class. Logistic regression has the highest recall (0.623819) but the lowest precision among these three models (0.267315), so it identifies more positive cases while also producing more false positives. The normal random forest has the highest precision (0.667590) but much lower recall (0.227788), indicating a more selective positive prediction pattern. The voting ensemble sits between them on precision (0.516058) and recall (0.440454), and has the highest test F1-score of the three (0.475268). These are fixed-threshold results; the precision-recall curves show how the tradeoff changes across score thresholds.

### Stored test-set evaluation

The following values are the existing final test evaluation output, not a new model-selection exercise:

| Model | Precision | Recall | F1-score | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|---:|
| Voting ensemble (tuned, balanced components) | 0.516058 | 0.440454 | 0.475268 | 0.794296 | 0.450399 |
| Decision tree (balanced, tuned) | 0.347393 | 0.560491 | 0.428933 | 0.753873 | 0.385916 |
| SVM (balanced, tuned) | 0.334090 | 0.556711 | 0.417582 | 0.771081 | 0.376199 |
| Logistic regression (balanced, tuned) | 0.267315 | 0.623819 | 0.374256 | 0.772193 | 0.409253 |
| Random forest (normal, tuned) | 0.667590 | 0.227788 | 0.339676 | 0.792955 | 0.443972 |

Test-set results were inspected during development. Consequently, this test set was not a completely untouched final holdout, and the test metrics should be interpreted as an evaluation of the finalized workflow rather than as a pristine unbiased estimate after one-time model selection.

## Visualizations

The existing notebooks contain:

- EDA plots for class balance and subscription rates across customer and campaign variables in [`01_exploration.ipynb`](notebooks/01_exploration.ipynb).
- A side-by-side final training-CV F1 versus average-precision bar plot in [`03_modeling.ipynb`](notebooks/03_modeling.ipynb).
- Shared-scale test-set confusion matrices and overlaid precision-recall curves with a positive-class prevalence baseline in [`03_modeling.ipynb`](notebooks/03_modeling.ipynb).

## Installation and execution

No dependency manifest is currently included in the repository. Create a Python environment, install the packages imported by the notebooks, and launch Jupyter:

```bash
python -m venv .venv
```

Activate the environment using the command appropriate for your operating system, then install the notebook dependencies:

```bash
python -m pip install numpy pandas scikit-learn matplotlib seaborn jupyter
jupyter notebook
```

Run the notebooks in order:

1. [`01_exploration.ipynb`](notebooks/01_exploration.ipynb)
2. [`02_preprocessing.ipynb`](notebooks/02_preprocessing.ipynb)
3. [`03_modeling.ipynb`](notebooks/03_modeling.ipynb)

The notebooks expect the repository-relative dataset path `data/bank/bank-full.csv`. The source modules are imported by the modeling workflow from `src/`.

## Repository structure

```text
.
|-- data/
|   `-- bank/
|       |-- bank-full.csv
|       |-- bank.csv
|       `-- bank-names.txt
|-- notebooks/
|   |-- 01_exploration.ipynb
|   |-- 02_preprocessing.ipynb
|   `-- 03_modeling.ipynb
|-- src/
|   |-- model_config.py
|   `-- preprocessing.py
|-- .gitignore
`-- README.md
```

## Limitations

- The dataset is observational; the EDA does not establish causal relationships.
- `duration` can create a deployment-timing limitation because it is measured during the call.
- Rare categories and high-contact-count groups may have fewer observations, so their observed rates can be unstable.
- The test set was inspected during development, so it is not a fully untouched final holdout.
- The reported precision-recall metric is scikit-learn `average_precision`, not an assertion that all PR-AUC definitions produce the same value.
