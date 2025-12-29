import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from imblearn.ensemble import BalancedRandomForestClassifier

# System Configuration
RANDOM_STATE = 42
REPLACEMENT_COST = 15000
INTERVENTION_COST = 2000
DATA_SOURCE = Path("../../data/Attrition.csv")
OUTPUT_PATH = Path("../../outputs/")

def engineer_risk_features(df):
    """Implement derived features based on organizational risk factors."""
    df['income_per_year'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1)
    df['overtime_dissatisfaction'] = (df['OverTime'] == 'Yes').astype(int) * (5 - df['JobSatisfaction'])
    return df

def get_optimized_threshold(y_true, y_probs):
    """Iterate through thresholds to maximize net financial profit."""
    best_profit = -np.inf
    opt_t = 0.5
    
    for t in np.linspace(0.1, 0.9, 90):
        y_pred = (y_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        # Profit = (TP * Net Saving) - (FP * Intervention Cost)
        current_profit = (tp * (REPLACEMENT_COST - INTERVENTION_COST)) - (fp * INTERVENTION_COST)
        
        if current_profit > best_profit:
            best_profit = current_profit
            opt_t = t
    return opt_t, best_profit

def run_production_pipeline():
    # Data Loading and Preparation
    path = DATA_SOURCE if DATA_SOURCE.exists() else Path("data/Attrition.csv")
    df = pd.read_csv(path)
    df['Attrition'] = (df['Attrition'] == 'Yes').astype(int)
    df = engineer_risk_features(df)
    
    X = df.drop(columns=['Attrition', 'EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], errors='ignore')
    y = df['Attrition']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    # Preprocessing
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object']).columns
    preprocessor = ColumnTransformer([
        ('scaler', StandardScaler(), num_cols),
        ('encoder', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    # Ensemble Model Training
    model = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(class_weight='balanced', solver='liblinear', random_state=RANDOM_STATE)),
            ('brf', BalancedRandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE))
        ],
        voting='soft'
    )

    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)
    
    model.fit(X_train_p, y_train)
    probs = model.predict_proba(X_test_p)[:, 1]

    # Optimization and Reporting
    threshold, profit = get_optimized_threshold(y_test, probs)
    preds = (probs >= threshold).astype(int)

    # Generate Confusion Matrix Visualization
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Optimized Decision Matrix (Threshold: {threshold:.2f})')
    
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH / 'final_optimized_cm.png')
    
    # Save Business Metrics
    report = pd.DataFrame({
        "Metric": ["Net Profit", "Recall", "Precision", "Opt Threshold"],
        "Value": [f"${profit:,.0f}", f"{recall_score(y_test, preds):.2%}", f"{precision_score(y_test, preds):.2%}", f"{threshold:.2f}"]
    })
    report.to_csv(OUTPUT_PATH / 'final_business_report.csv', index=False)
    print(f"Final Audit Complete. Projected Net Savings: ${profit:,.0f}")

if __name__ == "__main__":
    run_production_pipeline()