# 🚢 Titanic - Machine Learning from Disaster
![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Kaggle](https://img.shields.io/badge/Kaggle-Score%200.80861-success.svg)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)

An end-to-end Machine Learning pipeline developed for the classic [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic), achieving a **Top-tier accuracy score of 0.80861 (80.86%)** on the public test leaderboard.

---

## 📌 Project Overview
The objective of this project is to predict passenger survival on the Titanic using classification algorithms and advanced feature engineering techniques based on demographic, socio-economic, and family structure data.

## 🔑 Key Engineering Techniques & Highlights
* **Social Title Extraction (`Title`):** Extracted titles from names (*Mr*, *Mrs*, *Miss*, *Master*, *Rare*) to capture age-gender-status combinations.
* **Feature Categorization:** Binned continuous features like `Age` into age groups and `Fare` into quartiles (`FareGroup`) to minimize model noise.
* **Family Structure:** Combined `SibSp` and `Parch` to construct `FamilySize` and an `IsAlone` binary indicator.
* **Family Survival Tracking (`Family_Survival`):** Created a unique family identifier (`Surname_Fare`) to track historical survival rates among related group members, heavily boosting prediction power.

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Data Processing:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn` (`RandomForestClassifier`)

---

## 📊 Model Performance Evolution
| Version | Model / Technique | Kaggle Score |
| :--- | :--- | :--- |
| **V1** | Baseline Random Forest | `0.77751` |
| **V2** | Feature Engineering (Titles & Family Size) | `0.77751` |
| **V3** | XGBoost Classifier Experiment | `0.76794` |
| **V4** | Age & Fare Grouping | `0.78229` |
| **V5 (Final)** | **Family Survival Tracking + Random Forest** | **`0.80861`** 🎯 |

---

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/titanic-machine-learning.git](https://github.com/YOUR_USERNAME/titanic-machine-learning.git) ```
