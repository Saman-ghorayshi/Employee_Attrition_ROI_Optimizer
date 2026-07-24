import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import recall_score, f1_score, accuracy_score, confusion_matrix

# Suppression of convergence warnings for cleaner logs
warnings.filterwarnings('ignore')

# Global Parameters
UNIT_COST_ATTRITION = 15000
DATA_SOURCE = Path("../../data/Attrition.csv")
OUTPUT_DIR = Path("../../outputs/")

def load_and_preprocess():
    """Load data and perform feature selection/cleaning."""
    path = DATA_SOURCE if DATA_SOURCE.exists() else Path("data/Attrition.csv")
    df = pd.read_csv(path)
    
    # Drop invariant and identifier columns
    drop_list = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']
    df = df.drop(columns=[c for c in drop_list if c in df.columns], errors='ignore')
    
    # Encode target
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    return X, y

def get_transformation_pipeline(X):
    """Define the scaling and encoding logic."""
    num_vars = X.select_dtypes(include=['int64', 'float64']).columns
    cat_vars = X.select_dtypes(include=['object']).columns
    
    return ColumnTransformer([
        ('scaler', StandardScaler(), num_vars),
        ('encoder', OneHotEncoder(handle_unknown='ignore'), cat_vars)
    ])

def run_baseline_audit(X_train, X_test, y_train, y_test, preprocessor):
    """
    Evaluate performance on imbalanced data to demonstrate model failure.
    Corresponds to the 'Artelt Method' of failure reconstruction.
    """
    pipe = ImbPipeline([
        ('pre', preprocessor),
        ('clf', LogisticRegression(solver='liblinear', random_state=42))
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    
    print("\n--- Baseline Audit (Imbalanced) ---")
    print(f"Recall: {recall_score(y_test, preds):.4f}")
    print(f"F1-Score: {f1_score(y_test, preds):.4f}")
    return recall_score(y_test, preds)

def execute_model_competition(X_train, y_train, X_test, y_test, preprocessor):
    """GridSearch across multiple estimators using SMOTE resampled training sets."""
    param_grids = {
        'LogisticRegression': {
            'model': LogisticRegression(solver='liblinear', random_state=42),
            'params': {'clf__C': [0.1, 1, 10]}
        },
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {'clf__n_estimators': [100], 'clf__max_depth': [10, 20]}
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {'clf__learning_rate': [0.1], 'clf__n_estimators': [100]}
        }
    }
    
    results = []
    best_recall = 0
    winner_preds = None

    print("\nStarting optimization competition...")
    for name, config in param_grids.items():
        # Pipeline ensures SMOTE is only applied to training folds
        pipe = ImbPipeline([
            ('pre', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('clf', config['model'])
        ])
        
        search = GridSearchCV(pipe, config['params'], cv=5, scoring='recall')
        search.fit(X_train, y_train)
        
        preds = search.best_estimator_.predict(X_test)
        rec = recall_score(y_test, preds)
        
        results.append({
            'Model': name,
            'Recall': rec,
            'F1': f1_score(y_test, preds),
            'Accuracy': accuracy_score(y_test, preds)
        })
        
        if rec > best_recall:
            best_recall = rec
            winner_preds = preds

    return pd.DataFrame(results), winner_preds

def main():
    X, y = load_and_preprocess()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    preprocessor = get_transformation_pipeline(X)
    
    # Phase 1: Demonstrate Baseline Failure
    base_recall = run_baseline_audit(X_train, X_test, y_train, y_test, preprocessor)
    
    # Phase 2: Competitive Optimization
    comparison_df, winner_preds = execute_model_competition(X_train, y_train, X_test, y_test, preprocessor)
    
    print("\nOptimization Results:")
    print(comparison_df.sort_values(by='Recall', ascending=False).to_string(index=False))

    # Phase 3: ROI Calculation
    tn, fp, fn, tp = confusion_matrix(y_test, winner_preds).ravel()
    potential_savings = tp * UNIT_COST_ATTRITION
    
    print("\n--- Business Impact Analysis ---")
    print(f"Correct Identifications (TP): {tp}")
    print(f"False Negatives (Missed): {fn}")
    print(f"Projected Annual Savings: ${potential_savings:,.0f}")

    # Export results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(OUTPUT_DIR / "model_audit_comparison.csv", index=False)

if __name__ == "__main__":
    main()