# Finding Donors for Charity 🎁📊

An end-to-end Machine Learning project designed to employ several supervised algorithms to accurately model individuals' income using data collected from the 1994 U.S. Census. The primary goal is to help a non-profit organization identify potential donors who are most likely to contribute to charity (individuals earning more than $50,000 annually).

---

## 🚀 Project Overview
Non-profit organizations depend heavily on donations to survive. Deployed effectively, Machine Learning can help predict which individuals are most likely to become donors, reducing marketing costs and maximizing outreach efficiency. 

This project explores census data, applies rigorous data preprocessing, evaluates multiple supervised learning models, and optimizes the best-performing model to achieve high predictive accuracy and F-beta score.

---

## 🛠️ Tech Stack & Skills Demonstrated
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
* **Core ML Concepts:** 
  * Data Exploration & Visualization
  * Feature Transformation (Logarithmic transformation for skewed data)
  * Data Normalization & Scaling (`MinMaxScaler`)
  * One-Hot Encoding for Categorical Variables
  * Model Evaluation Metrics (Accuracy, Precision, Recall, F-Beta Score)
  * Model Tuning & Optimization (`GridSearchCV`)
  * Feature Importance Analysis

---

## 📊 Pipeline & Methodology

### 1. Data Exploration & Preprocessing
* **Exploration:** Analyzed the dataset to find the total number of records, individuals making $>50K$ vs $\le50K$, and checking for missing values.
* **Handling Skewed Features:** Applied a logarithmic transformation to highly skewed continuous features (like `capital-gain` and `capital-loss`).
* **Feature Scaling:** Normalized numerical features using a min-max scaler to ensure all features contribute equally to the model.
* **Encoding:** Converted categorical attributes (e.g., education, marital status) into numerical values using One-Hot Encoding.

### 2. Model Evaluation
Three different supervised learning algorithms were evaluated and compared against a naive predictor baseline:
1. **Decision Trees**
2. **Support Vector Machines (SVM)**
3. **Ensemble Methods (e.g., AdaBoost / Gradient Boosting)**

### 3. Model Tuning & Optimization
The best-performing model was chosen based on training time, accuracy, and **F-beta score** (which weights precision more heavily than recall since false positives cost the charity money). `GridSearchCV` was implemented to tune the hyperparameters of the chosen model.

### 4. Feature Importance
Extracted and visualized the top 5 most predictive features to understand what factors (like age, capital gain, or education) heavily influence an individual's income level.

---

## 📂 Repository Structure
```text
├── index.html              # Interactive HTML project report (Live Demo)
└── README.md               # Project documentation
