# Task 1: Titanic Data Cleaning and Preprocessing

A comprehensive, production-grade data cleaning, exploration, and preprocessing pipeline for the famous **Titanic Dataset**. This project implements machine learning best practices to handle missing values, cap outliers, engineer features, encode categories, and standardize data without introducing data leakage.

---

## 📂 Project Structure

```text
Task-1-Data-Cleaning-Preprocessing/
│
├── dataset/
│   └── Titanic-Dataset.csv        # Raw dataset from Kaggle
│
├── notebooks/
│   └── preprocessing.ipynb        # Jupyter Notebook with step-by-step pipeline execution
│
├── images/
│   ├── missing_values.png         # Diagnostic plot: count of missing values per feature
│   ├── boxplot_before.png         # Diagnostic plot: Age & Fare outlier distribution (Before Capping)
│   ├── boxplot_after.png          # Diagnostic plot: Age & Fare outlier distribution (After Capping)
│   └── correlation_heatmap.png    # Diagnostic plot: Heatmap of engineered feature correlations
│
├── cleaned_data/
│   ├── train_cleaned.csv          # Standard-scaled, leak-free training dataset split (80%)
│   ├── val_cleaned.csv            # Standard-scaled, leak-free validation dataset split (20%)
│   └── titanic_cleaned.csv        # Combined preprocessed model-ready dataset
│
├── requirements.txt               # Python package dependencies
├── README.md                      # Documentation (this file)
└── preprocessing.py               # Automated modular pipeline execution script
```

---

## 🛠️ Installation & Setup

1. **Navigate to the Directory**:
   ```bash
   cd Task-1-Data-Cleaning-Preprocessing
   ```

2. **Create a Virtual Environment** (Recommended):
   ```bash
   python -m venv .venv
   ```

3. **Activate the Virtual Environment**:
   * **Windows (PowerShell)**:
     ```bash
     .venv\Scripts\activate
     ```
   * **macOS / Linux**:
     ```bash
     source .venv/bin/activate
     ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ Preprocessing Workflow & ML Best Practices

This pipeline is designed defensively to align with industry-level machine learning standards:

### 1. Data Hygiene & Duplicate Handling
- Checks for duplicate rows using `.duplicated().sum()`. If duplicates are found, they are automatically dropped to prevent model bias.

### 2. Preventing Data Leakage
- **Stratified Train-Test Split**: The raw dataset is split into an **80% Train set** and a **20% Validation set** before any imputation, scaling, or statistics are computed.
- **Fitting Parameters**: Statistical fits (e.g. median ages, modes, scaler mean/variance, outlier thresholds) are calculated **only** on the training split.
- **Transformation**: Both training and validation splits are transformed independently using the parameters fitted from the training split. This guarantees that no validation information leaks into the features.

### 3. Missing Value Imputation
- **Age**: Imputed group-wise using the **median age** of the passenger's class (`Pclass`) and gender (`Sex`) group.
- **Embarked**: Imputed with the mode (`'S'`) of the training set.
- **Cabin**: Because over 77% of values are missing, the column is filled with `'Unknown'` and a binary indicator `HasCabin` (1/0) is created to retain survivability correlation.

### 4. Outlier Treatment (IQR Capping)
- Outliers in `Age` and `Fare` are identified using the Interquartile Range (IQR) method on the training set.
- Fares above `65.63` and Ages above `57.75` are capped (clipped) at the boundaries rather than deleted, which retains valuable records in a small dataset.

### 5. Feature Engineering
- **Title Extraction**: Passenger titles are extracted from the `Name` column and standardized into common groups (`Mr`, `Mrs`, `Miss`, `Master`, `Rare`).
- **Family Size**: Combines siblings/spouses and parents/children (`FamilySize = SibSp + Parch + 1`).
- **Is Alone**: Created as a binary indicator (1 if `FamilySize == 1`, else 0).

### 6. Encoding & Scaling
- One-hot encoding is applied to categorical variables (`Sex`, `Embarked`, `Title`) with `drop_first=True` to prevent the dummy variable trap.
- Numerical features (`Age`, `Fare`, `FamilySize`) are Z-score standardized using a `StandardScaler`.

---

## 🚀 How to Run the Pipeline

To execute the data cleaning pipeline and regenerate the outputs, run:
```bash
python preprocessing.py
```

Upon successful run, the script outputs the following metrics to the console using python's `logging` library:
- Shape and check for duplicate rows.
- Group medians and outlier boundaries.
- Cleaned dataset shape.
- Saved file logs.

---

## 🔮 Future Improvements

1. **Cross-Validation Pipeline**: Integrate the preprocessing steps directly into a Scikit-Learn `Pipeline` or `ColumnTransformer` to automate preprocessing during cross-validation.
2. **Robust Scaler**: Use `RobustScaler` instead of `StandardScaler` for the highly skewed `Fare` column to limit Z-score sensitivity to outliers.
3. **Advanced Imputation**: Use KNN Imputer or iterative imputers (MICE) to predict missing Age values instead of group-wise medians.
4. **Deck Extraction**: Extract the deck letter from the Cabin column to capture spatial survival correlation.
