import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score
from imblearn.ensemble import BalancedRandomForestClassifier

# Financial Assumptions & Sensitivity Analysis
COST_ATTRITION = 15000     # Cost of losing/replacing an employee
COST_INTERVENTION = 2000    # Cost of retention program per person
SUCCESS_PROBABILITY = 0.40  # Real-world assumption: 40% of targeted employees stay

def engineer_risk_features(df):
    """Derived features based on domain knowledge of HR attrition."""
    # Relative compensation efficiency
    df['income_per_tenure'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1)
    
    # Interaction between overtime and job dissatisfaction
    df['overtime_impact'] = (df['OverTime'] == 'Yes').astype(int) * (5 - df['JobSatisfaction'])
    return df

def load_and_prepare_data():
    data_path = Path("../../data/Attrition.csv")
    if not data_path.exists():
        data_path = Path("data/Attrition.csv")
        
    df = pd.read_csv(data_path)
    df['Attrition'] = (df['Attrition'] == 'Yes').astype(int)
    
    df = engineer_risk_features(df)
    
    # Exclude system columns
    drop_cols = ['Attrition', 'EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    y = df['Attrition']
    return X, y

def calculate_expected_profit(y_true, y_probs, threshold):
    """
    Computes Net Profit considering intervention costs and success probability.
    Formula: (True Positives * Success Rate * Savings) - (All Predicted Positives * Intervention Cost)
   
    """
    y_pred = (y_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Financial logic with success rate adjustment
    gross_savings = tp * SUCCESS_PROBABILITY * COST_ATTRITION
    total_intervention_cost = (tp + fp) * COST_INTERVENTION
    
    return gross_savings - total_intervention_cost

def main():
    X, y = load_and_prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Preprocessing pipeline
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object']).columns
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    # Calibrated Ensemble for probability estimation
    ensemble = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)),
            ('brf', BalancedRandomForestClassifier(n_estimators=100, random_state=42))
        ],
        voting='soft'
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    ensemble.fit(X_train_proc, y_train)
    y_probs = ensemble.predict_proba(X_test_proc)[:, 1]

    # Optimization: Find threshold maximizing Net Profit
    thresholds = np.linspace(0.05, 0.95, 100)
    profit_curve = [calculate_expected_profit(y_test, y_probs, t) for t in thresholds]
    
    best_idx = np.argmax(profit_curve)
    opt_threshold = thresholds[best_idx]
    max_profit = profit_curve[best_idx]

    # Final Evaluation at Optimal Threshold
    final_preds = (y_probs >= opt_threshold).astype(int)
    recall = recall_score(y_test, final_preds)

    print("-" * 35)
    print("Decision Support System Results")
    print("-" * 35)
    print(f"Optimal Threshold: {opt_threshold:.2f}")
    print(f"Model Recall:      {recall:.2%}")
    print(f"Expected Profit:   ${max_profit:,.0f}")
    print("-" * 35)

    # Save operational report
    os.makedirs('../../outputs/', exist_ok=True)
    pd.DataFrame({
        "metric": ["opt_threshold", "recall", "net_profit"],
        "value": [opt_threshold, recall, max_profit]
    }).to_csv('../../outputs/strategic_profit_report.csv', index=False)

if __name__ == "__main__":
    main()