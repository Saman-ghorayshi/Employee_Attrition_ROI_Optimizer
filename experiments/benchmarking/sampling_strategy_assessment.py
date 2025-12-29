import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, confusion_matrix, precision_recall_curve
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline

# Operational Constants
SEED = 42
UNIT_SAVINGS = 15000  # Estimated cost per avoided attrition
MIN_RECALL_TARGET = 0.75 # Threshold tuning floor

def load_processed_data():
    """Load and perform initial binary encoding for target."""
    data_path = Path("../../data/Attrition.csv")
    if not data_path.exists():
        data_path = Path("data/Attrition.csv")
        
    df = pd.read_csv(data_path)
    df['Attrition'] = (df['Attrition'] == 'Yes').astype(int)
    
    # Exclude non-informative features
    X = df.drop(columns=['Attrition', 'EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], errors='ignore')
    y = df['Attrition']
    return X, y

def get_preprocessor(X):
    """Encapsulate preprocessing logic for pipeline reuse."""
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object']).columns
    
    return ColumnTransformer([
        ('scaler', StandardScaler(), num_cols),
        ('encoder', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

def tune_decision_threshold(estimator, X_test, y_test):
    """Find optimal threshold to balance F1-score with a recall floor."""
    probs = estimator.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    
    # Selection logic: prioritize F1 within the recall constraint
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    valid_indices = np.where(recall >= MIN_RECALL_TARGET)[0]
    if len(valid_indices) == 0:
        return 0.5 # Fallback
    
    best_idx = valid_indices[np.argmax(f1_scores[valid_indices])]
    # Precision_recall_curve returns thresholds with length -1
    return thresholds[min(best_idx, len(thresholds)-1)]

def run_strategy_comparison():
    X, y = load_processed_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    
    preprocessor = get_preprocessor(X)

    # Define candidate strategies for handling imbalance
    experimental_set = {
        "smote_logit": ImbPipeline([
            ('pre', preprocessor), ('resampler', SMOTE(random_state=SEED)),
            ('clf', LogisticRegression(solver='liblinear', random_state=SEED))
        ]),
        "cost_sensitive_logit": ImbPipeline([
            ('pre', preprocessor),
            ('clf', LogisticRegression(class_weight='balanced', solver='liblinear', random_state=SEED))
        ]),
        "balanced_rf": ImbPipeline([
            ('pre', preprocessor),
            ('clf', BalancedRandomForestClassifier(n_estimators=100, random_state=SEED))
        ]),
        "hybrid_resampling": ImbPipeline([
            ('pre', preprocessor), ('resampler', SMOTETomek(random_state=SEED)),
            ('clf', LogisticRegression(solver='liblinear', random_state=SEED))
        ])
    }

    results = []
    print(f"Executing strategy benchmarking (Target Recall >= {MIN_RECALL_TARGET})...")

    for name, pipeline in experimental_set.items():
        pipeline.fit(X_train, y_train)
        
        # Baseline prediction (default 0.5)
        std_recall = recall_score(y_test, pipeline.predict(X_test))
        
        # Optimized prediction (threshold tuning)
        opt_threshold = tune_decision_threshold(pipeline, X_test, y_test)
        test_probs = pipeline.predict_proba(X_test)[:, 1]
        opt_preds = (test_probs >= opt_threshold).astype(int)
        opt_recall = recall_score(y_test, opt_preds)
        
        results.append({
            "Strategy": name,
            "Base_Recall": std_recall,
            "Opt_Recall": opt_recall,
            "Threshold": opt_threshold
        })

    summary_df = pd.DataFrame(results).sort_values(by="Opt_Recall", ascending=False)
    print("\nBenchmark Summary:")
    print(summary_df.to_string(index=False))

    # Business impact of the lead strategy
    best_case = summary_df.iloc[0]
    best_pipe = experimental_set[best_case['Strategy']]
    
    final_probs = best_pipe.predict_proba(X_test)[:, 1]
    final_preds = (final_probs >= best_case['Threshold']).astype(int)
    tp = confusion_matrix(y_test, final_preds)[1, 1]
    
    print(f"\nLead Strategy Financials ({best_case['Strategy']}):")
    print(f" Identified Leavers: {tp}")
    print(f" Projected Gross Savings: ${tp * UNIT_SAVINGS:,.0f}")

    # Persistence
    os.makedirs('../../outputs/', exist_ok=True)
    summary_df.to_csv('../../outputs/sampling_performance_audit.csv', index=False)

if __name__ == "__main__":
    run_strategy_comparison()