#!/usr/bin/env python3
import argparse, json, warnings
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--output', default='results')
    args = parser.parse_args()
    output_dir = Path(args.output); output_dir.mkdir(parents=True, exist_ok=True)
    expr = pd.read_csv(args.data, index_col=0)
    labels = pd.read_csv(args.labels, index_col=0)
    common = expr.index.intersection(labels.index)
    X = expr.loc[common].values
    le = LabelEncoder()
    y = le.fit_transform(labels.loc[common].iloc[:, 0])
    vt = VarianceThreshold(threshold=0.01)
    X = vt.fit_transform(X)
    selector = SelectKBest(mutual_info_classif, k=min(500, X.shape[1]))
    X = selector.fit_transform(X, y)
    classifiers = {
        'RandomForest': RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(n_estimators=300, random_state=42, eval_metric='logloss'),
        'SVM_RBF': SVC(kernel='rbf', probability=True, random_state=42),
        'LogReg': LogisticRegression(max_iter=1000, random_state=42),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    for name, clf in classifiers.items():
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
        scores = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc_ovr')
        results[name] = {'mean_auc': float(np.mean(scores)), 'std_auc': float(np.std(scores))}
        print(f'{name}: AUC = {np.mean(scores):.4f} +/- {np.std(scores):.4f}')
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('Classification complete.')

if __name__ == '__main__':
    main()
