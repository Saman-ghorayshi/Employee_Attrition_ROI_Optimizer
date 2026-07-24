import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Plot Configuration
FEATURES = ['MonthlyIncome', 'TotalWorkingYears']
OUTPUT_FILE = Path("../../outputs/decision_boundary_plot.png")
DATA_SOURCE = Path("../../data/Attrition.csv")

def load_and_scale():
    path = DATA_SOURCE if DATA_SOURCE.exists() else Path("data/Attrition.csv")
    df = pd.read_csv(path)
    
    # Binary encoding and feature selection
    df['Attrition'] = (df['Attrition'] == 'Yes').astype(int)
    X = df[FEATURES].values
    y = df['Attrition'].values
    
    # Scale and balance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, y)
    
    return X_res, y_res, scaler

def plot_decision_boundary(X, y, model):
    """Generate a 2D meshgrid to visualize class separation."""
    h = .02  # mesh step size
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 7))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.3)
    
    # Technical scatter plot
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('Monthly Income (Standardized)')
    plt.ylabel('Total Working Years (Standardized)')
    plt.title('Logistic Regression Decision Boundary (SMOTE Applied)')
    
    # Ensure output directory exists
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_FILE)
    print(f"Visual audit saved to: {OUTPUT_FILE}")

def main():
    X, y, _ = load_and_scale()
    
    # Simplified 2D model for visual diagnostics
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X, y)
    
    plot_decision_boundary(X, y, model)

if __name__ == "__main__":
    main()