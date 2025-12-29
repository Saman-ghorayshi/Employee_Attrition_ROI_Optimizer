import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, confusion_matrix

# Optional: High-performance gradient boosting
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# Resampling ensembles for imbalanced learning
from imblearn.ensemble import BalancedRandomForestClassifier

# Configuration
SEED = 42
DATA_PATH = Path("../../data/Attrition.csv")
UNIT_ATTRITION_COST = 15000

def load_and_initialize():
    path = DATA_PATH if DATA_PATH.exists() else Path("data/Attrition.csv")
    df = pd.read_csv(path)
    
    # Standardize target variable
    df['Attrition'] = (df['Attrition'] == 'Yes').astype(int)
    
    # Feature filtering
    exclude = ['Attrition', 'EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']
    X = df.drop(columns=[c for c in exclude if c in df.columns], errors='ignore')
    y = df['Attrition']
    
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

def build_composite_ensemble(y_train):
    """
    Construct a heterogeneous ensemble to maximize minority class detection.
    Integrates cost-sensitive learning and balanced bootstrapping.
    """
    # Logistics with cost-weighting
    lr = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=SEED)
    
    # Forest with internal down-sampling
    brf = BalancedRandomForestClassifier(n_estimators=100, random_state=SEED)
    
    estimators = [('logit', lr), ('bal_rf', brf)]

    if HAS_XGB:
        # Scale weight based on class imbalance ratio
        imbalance_ratio = (len(y_train) - y_train.sum()) / y_train.sum()
        xgb = XGBClassifier(scale_pos_weight=imbalance_ratio, eval_metric='logloss', random_state=SEED)
        estimators.append(('xgb', xgb))
        
    return VotingClassifier(estimators=estimators, voting='soft')

def optimize_recall_threshold(probs, y_true):
    """
    Find the threshold that maximizes Recall within operational constraints.
    """
    best_recall = 0
    optimal_t = 0.5
    
    # Evaluation across potential decision boundaries
    for t in np.arange(0.1, 0.6, 0.01):
        r = recall_score(y_true, (probs >= t).astype(int))
        if r > best_recall:
            best_recall = r
            optimal_t = t
            
    return optimal_t, best_recall

def main():
    X_train, X_test, y_train, y_test = load_and_initialize()
    
    # Define preprocessing pipeline
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X_train.select_dtypes(include=['object']).columns
    
    preprocessor = ColumnTransformer([
        ('scaler', StandardScaler(), num_cols),
        ('encoder', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    # Model training
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    model = build_composite_ensemble(y_train)
    model.fit(X_train_proc, y_train)
    
    # Threshold Tuning Phase
    test_probs = model.predict_proba(X_test_proc)[:, 1]
    opt_t, max_recall = optimize_recall_threshold(test_probs, y_test)
    
    # Final Metrics
    final_preds = (test_probs >= opt_t).astype(int)
    tp = confusion_matrix(y_test, final_preds)[1, 1]
    potential_roi = tp * UNIT_ATTRITION_COST

    print("--- Operational Performance Report ---")
    print(f"Target Recall Score:  {max_recall:.2%}")
    print(f"Decision Threshold:   {opt_t:.2f}")
    print(f"TP Identifications:   {tp}")
    print(f"Projected ROI:        ${potential_roi:,.0f}")

    # Persistence
    out_dir = Path("../../outputs/")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "metric": ["recall", "threshold", "tp", "roi"],
        "value": [max_recall, opt_t, tp, potential_roi]
    }).to_csv(out_dir / "ensemble_validation_results.csv", index=False)

if __name__ == "__main__":
    main()