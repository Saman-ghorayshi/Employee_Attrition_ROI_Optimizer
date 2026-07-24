import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.ensemble import BalancedRandomForestClassifier

# ignore  warnings so the terminal doesnt get spammed
warnings.filterwarnings('ignore') 

def main():
    print("Starting SHAP Explainability script...")
    
    # simple  path check avoid crash stuff
    file_path = "../../data/Attrition.csv"
    if not os.path.exists(file_path):
        file_path = "data/Attrition.csv"
        
    df = pd.read_csv(file_path)
    
    # preping target variable
    df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    y = df['Attrition']
    
    # drop useless stuff
    cols_to_drop = ['Attrition', 'EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # rebuild the column transformer from the main pipeline
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X_train.select_dtypes(include=['object']).columns
    
    ct = ColumnTransformer([
        ('scaler', StandardScaler(), num_cols),
        ('ohe', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])
    
    X_train_encoded = ct.fit_transform(X_train)
    X_test_encoded = ct.transform(X_test)
    
    # get the new column names because OHE creates a bunch of extra columns
    raw_names = ct.get_feature_names_out()
    clean_names = [n.replace('scaler__', '').replace('ohe__', '') for n in raw_names]
    
    # Retrain the ensemble model
    print("Training the voting classifier...")
    log_reg = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)
    brf = BalancedRandomForestClassifier(n_estimators=100, random_state=42)
    
    model = VotingClassifier(estimators=[('lr', log_reg), ('brf', brf)], voting='soft')
    model.fit(X_train_encoded, y_train)
    
    print("Calculating SHAP values (this might take a minute)...")
    
    # sample bg data cause KernelExplainer is super slow on full datasets :(
    background_data = shap.sample(X_train_encoded, 100, random_state=42)
    
    #  function to just get the probability of leaving 
    def predict_leave_prob(data):
        return model.predict_proba(data)[:, 1]
        
    explainer = shap.KernelExplainer(predict_leave_prob, background_data)
    
    # test  first 50 rows to save time
    X_test_sample = X_test_encoded[:50]
    shap_values = explainer.shap_values(X_test_sample)
    
    # make sure output directory is ther
    out_dir = "../../outputs"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    print("Saving global feature plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_sample, feature_names=clean_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'shap_global_summary.png'))
    plt.close()
    
    print("Saving waterfall plot for the first employee in the test set...")
    plt.figure(figsize=(8, 6))
    
    explanation = shap.Explanation(
        values=shap_values[0], 
        base_values=explainer.expected_value, 
        data=X_test_sample[0], 
        feature_names=clean_names
    )
    shap.plots.waterfall(explanation, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'shap_local_waterfall.png'))
    plt.close()
    
    print("Done! Check outputs folder.")

if __name__ == "__main__":
    main()
