# ML Gene Expression Classifier

> You have expression data for 200 COPD patients and 200 healthy controls. Can a machine learning model tell them apart from gene expression alone? And more importantly, which genes is it using? Those genes are your biomarker candidates.

## Why ML on Gene Expression Data

Differential expression finds genes that change on average between groups. Machine learning finds combinations of genes that predict group membership. A gene might not be significant on its own but could be highly predictive when combined with others. ML captures these combinatorial patterns that univariate tests miss.

## The Feature Selection Problem

20,000 genes and 100 samples: the classic p >> n problem. Without feature selection, you'll overfit spectacularly. This pipeline uses variance filtering followed by mutual information to get down to the 500 most informative genes.

## Four Classifiers, Head-to-Head

| Model | Strengths | Weaknesses |
|-------|-----------|------------|
| **Random Forest** | Handles high-D well | Can overfit with correlated features |
| **XGBoost** | Often top performer | Needs hyperparameter tuning |
| **SVM (RBF)** | Excellent for small n | No built-in feature importance |
| **Logistic Regression** | Interpretable | Assumes linear boundary |

All evaluated with **5-fold nested cross-validation**, the only honest way to report performance on gene expression data.

## Usage
```bash
python train_classifier.py --data data/expression.csv --labels data/labels.csv --output results/
```

## The Biomarker Discovery Angle

Once you've trained a good model, use SHAP values to identify which genes the model relies on most. Those top features are your computationally predicted biomarkers. Validate them in independent datasets and you've got the computational side of a biomarker paper.
