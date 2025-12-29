import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Analysis Parameters
K_OPTIMAL = 4
SEED = 42
DATA_PATH = Path("../../data/Attrition.csv")

def load_data():
    path = DATA_PATH if DATA_PATH.exists() else Path("data/Attrition.csv")
    df = pd.read_csv(path)
    # Target transformation for mean calculations
    df['attrition_val'] = (df['Attrition'] == 'Yes').astype(int)
    return df

def get_pipeline_preprocessor(df):
    """Define standard HR feature transformation logic."""
    X = df.drop(columns=['Attrition', 'attrition_val', 'EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], errors='ignore')
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    return ColumnTransformer([
        ('scaler', StandardScaler(), numeric_features),
        ('encoder', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]), X

def run_segmentation_audit():
    df = load_data()
    preprocessor, X = get_pipeline_preprocessor(df)
    
    # Process data and execute K-Means
    X_processed = preprocessor.fit_transform(X)
    
    kmeans = KMeans(n_clusters=K_OPTIMAL, n_init=10, random_state=SEED)
    df['cluster'] = kmeans.fit_predict(X_processed)

    # Generate Cluster Performance Table
    cluster_stats = df.groupby('cluster').agg({
        'attrition_val': ['count', 'mean']
    })
    
    # Flatten multi-index columns for readability
    cluster_stats.columns = ['total_employees', 'attrition_rate']
    cluster_stats['attrition_percentage'] = cluster_stats['attrition_rate'] * 100
    
    # Identify high-risk segment automatically
    high_risk_cluster = cluster_stats['attrition_rate'].idxmax()
    overall_avg = df['attrition_val'].mean()

    print("--- Segmentation Audit Results ---")
    print(f"Company Average Attrition: {overall_avg:.2%}")
    print("\nCluster Distribution:")
    print(cluster_stats.sort_values(by='attrition_rate', ascending=False).to_string())
    
    print(f"\nOperational Focus: Cluster {high_risk_cluster} identified as primary risk group.")
    
    # Persistence
    output_dir = Path("../../outputs/")
    output_dir.mkdir(parents=True, exist_ok=True)
    cluster_stats.to_csv(output_dir / "persona_segmentation_report.csv")

if __name__ == "__main__":
    run_segmentation_audit()