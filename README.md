# 📊 Employee Attrition ROI Optimizer

### **Course:** Computer Applications in Business Systems (کاربرد کامپیوتر در سیستم‌های تجاری)

**Professor:** [Dr. Fatemeh Elahi]

**Developed by:** [Hossein Abedzadeh] | [Meysam Najafifard] | [Saman Ghorayshi]


## 📝 Project Executive Summary

This project implements a **Decision Support System (DSS)** designed to predict employee attrition and optimize the financial impact of retention interventions. By integrating machine learning with financial logic, the system identifies high-risk employee "personas" and determines the optimal probability threshold to maximize Net Profit.

---

## 📂 Project Architecture

The repository follows a professional R&D-to-Production hierarchy, separating experimental research from operational logic.

```text
.
├── core_system/           # Operational Engine
│   └── production_engine.py
├── experiments/           # R&D Sandbox
│   ├── benchmarking/      # Model competition & Sensitivity Analysis
│   │   ├── model_benchmark.py
│   │   ├── sampling_strategy_assessment.py
│   │   ├── sensitivity_analysis_audit.py
│   │   └── run_full_analysis.py
│   ├── prototypes/        # Persona discovery & business logic
│   │   ├── 3_discovery_persona_analysis.py
│   │   ├── business_impact_optimizer.py
│   │   └── ensemble_risk_validator.py
│   └── visualizations/    # Process animations & Decision boundaries
│       ├── cluster_optimization_audit.py
│       ├── cluster_formation_viz.py
│       ├── smote_balancing_viz.py
│       └── logistic_boundary_viz.py
├── docs/                  # Technical audit & verification
│   └── model_verification_audit.py
├── data/                  # Source Dataset (Attrition.csv)
├── outputs/               # Generated Visual Assets & CSV Reports
├── init_project.py        # Environment setup script
└── run_all.py             # Full pipeline automation script

```

---

## 🔬 Technical Methodology

### 1. High-Dimensional Persona Extraction

Using **K-Means Clustering**, we identified 4 distinct employee segments. The "High-Risk" segment shows an attrition rate of **20.8%**.

* **35-Dimensional Space:** Features were expanded via One-Hot Encoding and normalized using `StandardScaler`.
* **Distance Metric:** We utilized **Euclidean Distance** to calculate similarities across all 35 dimensions, ensuring "neighborhood" accuracy for both clustering and SMOTE.

### 2. Strategic Model Selection: Logistic Regression (LR) vs. Random Forest (RF)

While Random Forest showed high potential in low-cost scenarios, **Logistic Regression** was selected as the champion model for the following business reasons:

* **Interpretability:** LR provides clear coefficients, allowing managers to understand *why* an employee is flagged (e.g., Overtime impact).
* **Robustness (Sensitivity Analysis):** Our tests showed that as intervention costs increase from $2,000 to $10,000, LR maintains higher profitability ($185k) compared to RF ($55k), which suffers from high False Positive costs.
* **Convex Optimization:** LR uses a **Convex Loss Function**, guaranteeing convergence to the global minimum, unlike non-convex complex models that may trap in local minima.

### 3. Class Imbalance (SMOTE)

To address the 16% minority class, we used **SMOTE** to generate synthetic samples. By interpolating between the 5 nearest neighbors in the 35-dimensional space, the model learns a more robust decision boundary.

---

## 💰 Business Impact & ROI

The system optimizes the decision threshold to maximize the following ROI formula:


* **Replacement Savings:** $15,000 (Cost of hiring/training).
* **Intervention Cost:** Variable ($2,000 - $10,000).
* **Max Profit Achieved:** ~$430,000 (Test set projection at $2k cost level).

---

## 🛡️ Technical Audit

The `docs/model_verification_audit.py` performs a manual mathematical check. We verified the model's output by manually calculating the **Sigmoid Function** , ensuring the software implementation aligns perfectly with statistical theory.

---

## 🚀 How to Run

1. **Setup:** Place `Attrition.csv` in the `/data` folder.
2. **Install Dependencies:** `pip install -r requirements.txt`
3. **Initialize:** `python init_project.py`
4. **Execute All:** `python run_all.py`

---

*This system demonstrates the application of advanced predictive analytics in modern Business Information Systems to drive data-led human capital strategy.*
## 🛠 AI Usage Disclosure
In alignment with modern industry standards, Large Language Models (LLMs) were utilized as strategic assistants in the following areas of this project:
- **Code Refactoring & Optimization:** To ensure the multi-folder architecture and relative pathing follow professional clean-code principles.
- **Mathematical Logic Validation:** To verify the manual calculation of the Sigmoid function against standard library outputs.
- **Documentation:** To assist in generating professional-grade technical documentation and data visualizations.
The core business logic, ROI formulas, and strategic decision-making (such as selecting Logistic Regression based on sensitivity analysis) were led by the project team.