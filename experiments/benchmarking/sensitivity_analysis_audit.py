import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# 1. تنظیم مسیرها (مطابق با ساختار درختی شما)
DATA_PATH = Path("../../data/Attrition.csv")
OUTPUT_CSV = Path("../../outputs/sensitivity_analysis_numbers.csv")
OUTPUT_PLOT = Path("../../outputs/profit_sensitivity_plot.png")

def run_sensitivity_analysis():
    # بارگذاری و پیش‌پردازش
    if not DATA_PATH.exists():
        print(f"Error: Data not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    df['Attrition'] = (df['Attrition'] == 'Yes').astype(int)
    X = df.drop(columns=['Attrition', 'EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], errors='ignore')
    y = df['Attrition']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), X.select_dtypes(include=['int64', 'float64']).columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), X.select_dtypes(include=['object']).columns)
    ])
    
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)
    
    # مدل‌ها
    models = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42)
    }
    
    intervention_costs = np.arange(2000, 11000, 1000)
    savings_per_person = 15000
    results_list = []

    print("\n" + "="*50)
    print(f"{'Cost ($)':<10} | {'Model':<20} | {'Max Profit ($)':<15}")
    print("-" * 50)

    for cost in intervention_costs:
        for name, model in models.items():
            model.fit(X_train_p, y_train)
            probs = model.predict_proba(X_test_p)[:, 1]
            
            best_profit = -np.inf
            for threshold in np.linspace(0.1, 0.9, 100):
                preds = (probs >= threshold).astype(int)
                tp = np.sum((preds == 1) & (y_test == 1))
                fp = np.sum((preds == 1) & (y_test == 0))
                profit = (tp * savings_per_person) - (fp * cost)
                if profit > best_profit:
                    best_profit = profit
            
            results_list.append({
                'Intervention_Cost': cost,
                'Model': name,
                'Max_Profit': best_profit
            })
            print(f"{cost:<10} | {name:<20} | {best_profit:<15,}")

    print("="*50)

    # ذخیره در CSV
    res_df = pd.DataFrame(results_list)
    res_df.to_csv(OUTPUT_CSV, index=False)

    # رسم نمودار
    plt.figure(figsize=(10, 6))
    for name in models:
        subset = res_df[res_df['Model'] == name]
        plt.plot(subset['Intervention_Cost'], subset['Max_Profit'], marker='o', label=name, linewidth=2)
    
    plt.title('Profit Sensitivity Analysis: Decision Support Report')
    plt.xlabel('Intervention Cost per Employee ($)')
    plt.ylabel('Max Potential Net Profit ($)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(OUTPUT_PLOT)
    
    print(f"\n✅ Numbers saved to: {OUTPUT_CSV}")
    print(f"✅ Plot saved to: {OUTPUT_PLOT}")

if __name__ == "__main__":
    run_sensitivity_analysis()