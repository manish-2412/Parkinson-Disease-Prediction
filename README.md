# Parkinson's Disease Prediction using Machine Learning

This project predicts if a person has Parkinson's disease by looking at their voice data. I used a model called SVM. I also used SMOTE to balance the data and GridSearchCV to find the best settings.

---

## Table of Contents

- [About the Project](#about-the-project)
- [Dataset](#dataset)
- [Steps I Followed](#steps-i-followed)
- [Results](#results)
- [Charts](#charts)
- [Tools Used](#tools-used)
- [How to Run](#how-to-run)
- [Project Files](#project-files)
- [Future Work](#future-work)
- [Author](#author)

---

## About the Project

Parkinson's disease can change the way a person speaks. In this project, I use numbers taken from voice recordings to decide if a person is **healthy (0)** or has **Parkinson's (1)**.

**What I wanted to do**

- Build a model that predicts Parkinson's well
- Fix the problem of having more samples of one group than the other
- Find which voice features are the most useful for the prediction

---

## Dataset

- **File:** `pd_speech_features.csv`
- **Source:** [UCI Machine Learning Repository: Parkinson's Disease Classification](https://archive.ics.uci.edu/dataset/470/parkinson+s+disease+classification)
- **Size:** [756] rows and [754] columns
- **Target column:** `class` (0 = Healthy, 1 = Parkinson's)

I removed the `id` column and any rows that had no value in the `class` column.

---

## Steps I Followed

1. **Clean the data:** Removed the `id` column and rows with a missing `class` value.
2. **Look at the data:** Checked for missing values and made a chart of how many people are in each group.
3. **Split the data:** Kept 80% for training and 20% for testing. I did this first so the test data stays untouched.
4. **Scale the features:** Used `StandardScaler` so all features are on the same scale. I fitted it on the training data only.
5. **Balance the data:** Used **SMOTE** on the training data to make both groups the same size.
6. **Train the model:** Used an **SVM** model and tried different settings (`C` and `gamma`) with **GridSearchCV** to find the best ones.
7. **Test the model:** Checked accuracy, confusion matrix, precision, recall, and F1-score on the test data.
8. **Find the top features:** Used **mutual information** to find the 10 most useful features and checked how they relate to each other.

---

## Results

| Measure | Score |
|---|---|
| Accuracy | [95]% |
| Precision | [94]% |
| Recall | [96]% |
| F1-score | [95]% |

**Best settings found:** `{'C': [..], 'gamma': '[..]', 'kernel': 'rbf'}`

## Charts

The notebook makes these charts:

- Number of people in each group, before and after SMOTE
- Top 10 most useful features
- Heatmap showing how the top 10 features relate to each other
- Chart showing how much each feature adds to the prediction

![Class Distribution](images/class_distribution.png)
![Feature Importance](images/feature_importance.png)

---

## Tools Used

- **Language:** Python
- **Data work:** Pandas, NumPy
- **Machine learning:** Scikit-learn (SVM, GridSearchCV, SelectKBest), imbalanced-learn (SMOTE)
- **Charts:** Matplotlib, Seaborn
- **Where I ran it:** Jupyter Notebook

---

## How to Run

1. **Download the project**

   ```bash
   git clone https://github.com/manish-2412/Parkinson-Disease-Prediction.git
   cd Parkinson-Disease-Prediction
   ```

2. **Install the libraries**

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
   ```

3. **Add the dataset**

   Put `pd_speech_features.csv` in the project folder.

4. **Open the notebook**

   ```bash
   jupyter notebook
   ```

   Then run all the cells.

---

## Project Files

```
Parkinson-Disease-Prediction
├── pd_speech_features.csv       # Data file
├── parkinsons_prediction.ipynb  # Main notebook
├── images/                      # Charts used in this README
└── README.md
```

---

## Future Work

- Try more models like Random Forest, XGBoost, and Logistic Regression
- Add statistical tests (t-test or Mann-Whitney) to check which features really matter
- Use a Pipeline so scaling and SMOTE happen safely inside cross-validation
- Try PCA to reduce the number of features
- Build a simple web app that gives predictions

---

## Author

**Manish Chandrasekar**

[LinkedIn](https://www.linkedin.com/in/manish-chandrasekar/) | [GitHub](https://github.com/manish-2412) | [Email](mailto:c.manish002@gmail.com)
