
---

# 🛡️ FraudShield: End-to-End Financial Fraud Detection

## 📌 Project Overview

This project implements a high-performance machine learning pipeline to detect fraudulent transactions in financial datasets. With fraud representing only **0.13%** of the data, the core challenge was addressing extreme class imbalance while maintaining low false-alarm rates for legitimate customers.

## 🚀 Key Technical Features

* **Engineered Features:** Created cyclical time features (Hour of Day) and "Net Change" balance features to capture the signature of "account draining" behavior.
* **Imbalance Handling:** Utilized `scale_pos_weight` and `stratified` sampling to ensure the model prioritizes catching the minority fraud class.
* **Hardware Optimized:** Configured for local execution on **NVIDIA RTX 3060** using the `hist` tree method for accelerated training.
* **Precision-Recall Tuning:** Custom threshold optimization to guarantee a **90% Recall** rate, ensuring 9 out of 10 frauds are caught.

---

## 🛠️ Data Engineering & Insights

Raw financial data is often redundant. My pipeline transforms bank snapshots into "behavioral signals":

1. **Time-Based Analysis:** Discovered that while total traffic drops at 4:00 AM, the fraud rate spikes to **22%**, identifying nighttime as a high-risk window.
2. **Balance Differentials:** Calculated the difference between `oldbalance` and `newbalance`.
* **Insight:** Fraudsters often "drain" accounts to exactly zero. My engineered `balance_diff_orig` feature flags this "perfect match" signature.


3. **Feature Reduction:** Removed perfectly correlated features (1.00 correlation) to eliminate multicollinearity and reduce the memory footprint by **30%**.

---

## 🤖 Model Architecture: XGBoost

I implemented a tuned **XGBoost Classifier** with the following production-grade configurations:

* **Complexity Control:** Set `max_depth=4` and `min_child_weight=50` to prevent the model from "hallucinating" patterns based on random noise.
* **Learning Pace:** Applied a slow `learning_rate` (0.05) with 200 estimators for stable convergence.
* **Hardware Acceleration:** Leveraged `tree_method='hist'` to maximize GPU/Parallel CPU efficiency.

---

## 📊 Performance & Evaluation

Standard accuracy is a trap for fraud detection. I evaluated the model using:

* **Precision-Recall Curve:** To find the optimal business threshold.
* **Robust Scaling:** Used `RobustScaler` to ensure the model isn't biased by high-value outlier transactions (up to $90M).

| Metric | Goal | Result |
| --- | --- | --- |
| **Recall** | > 90% | Successfully caught 90%+ of fraud cases |
| **Threshold** | Optimized | Tuned to 0.15 - 0.20 for maximum precision |

---



---

## 💻 Tech Stack

* **Language:** Python
* **ML Libraries:** XGBoost, Scikit-Learn
* **Analysis:** Pandas, NumPy, Matplotlib, Seaborn
* **Infrastructure:** Linux/Ubuntu, NVIDIA RTX 3060

---
