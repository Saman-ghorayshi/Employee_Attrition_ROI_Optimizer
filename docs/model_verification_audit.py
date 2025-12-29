import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Attrition.csv"

def run_diagnostic_audit():
    # added dynamic path to make sure it works
    if not DATA_PATH.exists():
        print(f"Error: Data source not found at {DATA_PATH}")
        return
    
    df = pd.read_csv(DATA_PATH)
    df['Attrition'] = (df['Attrition'] == 'Yes').astype(int)
    X = df.drop(columns=['Attrition', 'EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], errors='ignore')
    y = df['Attrition']

    # 1. K-Means Verification
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object']).columns
    pre = ColumnTransformer([('s', StandardScaler(), num_cols), ('e', OneHotEncoder(), cat_cols)])
    X_p = pre.fit_transform(X)
    
    km = KMeans(n_clusters=4, n_init=10, random_state=42)
    km.fit(X_p)
    print(f"Cluster Inertia (WCSS): {km.inertia_:.2f}")

    # 2. SMOTE Verification
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_p, y)
    print(f"Resampling Audit: Original Class Ratio {y.mean():.2%}, Balanced Ratio {y_res.mean():.2%}")

    # 3. Logistic Regression & Sigmoid Simulation
    clf = LogisticRegression(solver='liblinear', random_state=42)
    clf.fit(X_res, y_res)
    
    # Mathematical Integrity Check for a sample observation
    sample_idx = 25
    x_sample = X_res[sample_idx]
    
    # Calculate score: z = wX + b
    z = np.dot(clf.coef_[0], x_sample) + clf.intercept_[0]
    prob_manual = 1 / (1 + np.exp(-z))
    prob_model = clf.predict_proba(x_sample.reshape(1, -1))[0, 1]
    
    print("\n--- Technical Integrity Report ---")
    print(f"Manual Sigmoid Calculation: {prob_manual:.4f}")
    print(f"Scikit-Learn Output:      {prob_model:.4f}")
    print(f"Verification Status:       {'PASS' if np.isclose(prob_manual, prob_model) else 'FAIL'}")

if __name__ == "__main__":
    run_diagnostic_audit()
