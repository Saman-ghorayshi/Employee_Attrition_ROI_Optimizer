import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

def run_diagnostic_audit():
    data_path = Path("../../data/Attrition.csv")
    if not data_path.exists(): data_path = Path("data/Attrition.csv")
    
    df = pd.read_csv(data_path)
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

    # 3. LogReg Weights & Sigmoid Simulation
    clf = LogisticRegression(solver='liblinear', random_state=42)
    clf.fit(X_res, y_res)
    
    # Manual Sigmoid Verification for Observation #25
    sample_idx = 25
    x_sample = X_res[sample_idx]
    
    # Mathematical score: z = wX + b
    z = np.dot(clf.coef_[0], x_sample) + clf.intercept_[0]
    prob_manual = 1 / (1 + np.exp(-z))
    prob_model = clf.predict_proba(x_sample.reshape(1, -1))[0, 1]
    
    print("\n--- Mathematical Integrity Check ---")
    print(f"Manual Calculation S(z): {prob_manual:.4f}")
    print(f"Model predict_proba:     {prob_model:.4f}")
    print(f"Integrity Status:        {'PASS' if np.isclose(prob_manual, prob_model) else 'FAIL'}")

if __name__ == "__main__":
    run_diagnostic_audit()