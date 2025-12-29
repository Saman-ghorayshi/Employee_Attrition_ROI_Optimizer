import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import recall_score, f1_score, accuracy_score

# Configuration
RANDOM_STATE = 42
DATA_PATH = Path("../../data/Attrition.csv")
ATTRITION_COST = 15000 

def get_data_source():
    if not DATA_PATH.exists():
        return Path("data/Attrition.csv")
    return DATA_PATH

def load_dataset():
    path = get_data_source()
    df = pd.read_csv(path)
    
    # Drop constants and high-cardinality identifiers
    df = df.drop(columns=['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], errors='ignore')
    df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    return X, y, df

def execute_benchmarking():
    X, y, df_raw = load_dataset()
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Unsupervised Clustering (Segment Analysis)
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
    
    X_transformed = preprocessor.fit_transform(X)
    
    # K=4 based on business interpretability
    kmeans = KMeans(n_clusters=4, n_init=10, random_state=RANDOM_STATE)
    df_raw['Cluster'] = kmeans.fit_predict(X_transformed)
    
    print("Cluster Distribution & Attrition Rates:")
    for i in range(4):
        segment = df_raw[df_raw['Cluster'] == i]
        rate = segment['Attrition'].mean()
        print(f" Group {i}: n={len(segment)}, Rate={rate:.2%}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    # Model Pipeline Definitions
    models = {
        'logit_regression': {
            'estimator': LogisticRegression(solver='liblinear', random_state=RANDOM_STATE),
            'grid': {'clf__C': [0.1, 1, 10], 'clf__penalty': ['l1', 'l2']}
        },
        'random_forest': {
            'estimator': RandomForestClassifier(random_state=RANDOM_STATE),
            'grid': {'clf__n_estimators': [100, 200], 'clf__max_depth': [10, None]}
        },
        'svc': {
            'estimator': SVC(probability=True, random_state=RANDOM_STATE),
            'grid': {'clf__C': [1, 10], 'clf__kernel': ['rbf']}
        }
    }

    results = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print("\nStarting Model Cross-Validation...")
    for name, cfg in models.items():
        # Correctly using ImbPipeline to avoid leakage
        pipeline = ImbPipeline([
            ('pre', preprocessor),
            ('smote', SMOTE(random_state=RANDOM_STATE)),
            ('clf', cfg['estimator'])
        ])
        
        search = GridSearchCV(pipeline, cfg['grid'], cv=skf, scoring='recall', n_jobs=-1)
        search.fit(X_train, y_train)
        
        y_pred = search.best_estimator_.predict(X_test)
        
        results.append({
            'model': name,
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'accuracy': accuracy_score(y_test, y_pred)
        })

    # Summary Report
    performance_df = pd.DataFrame(results).sort_values(by='recall', ascending=False)
    print("\n" + "-"*40)
    print(performance_df.to_string(index=False))
    print("-"*40)

    # Financial Impact (Winner)
    top_recall = performance_df.iloc[0]['recall']
    total_leavers = y_test.sum()
    identified = int(top_recall * total_leavers)
    savings = identified * ATTRITION_COST
    
    print(f"\nFinancial Analysis ({performance_df.iloc[0]['model']}):")
    print(f" Identified Leavers: {identified}/{total_leavers}")
    print(f" Potential Savings: ${savings:,.0f}")

    # Output Persistence
    os.makedirs('../../outputs/', exist_ok=True)
    performance_df.to_csv('../../outputs/benchmark_results.csv', index=False)

if __name__ == "__main__":
    execute_benchmarking()