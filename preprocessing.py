import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Set paths dynamically relative to the script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
BASE_DIR = SCRIPT_DIR if os.path.basename(SCRIPT_DIR) == 'Task-1-Data-Cleaning-Preprocessing' else os.path.join(SCRIPT_DIR, 'Task-1-Data-Cleaning-Preprocessing')

DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'Titanic-Dataset.csv')
IMAGES_DIR = os.path.join(BASE_DIR, 'images')
CLEANED_DATA_DIR = os.path.join(BASE_DIR, 'cleaned_data')
CLEANED_DATA_PATH = os.path.join(CLEANED_DATA_DIR, 'titanic_cleaned.csv')

# Ensure directories exist
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(CLEANED_DATA_DIR, exist_ok=True)

# 1. Load the dataset
print("Loading Titanic Dataset...")
df = pd.read_csv(DATASET_PATH)
print(f"Dataset shape: {df.shape}")

# 2. Visualize Missing Values
print("Generating missing values plot...")
plt.figure(figsize=(10, 6))
missing_data = df.isnull().sum()
missing_data = missing_data[missing_data > 0]
if not missing_data.empty:
    missing_data = missing_data.sort_values(ascending=False)
    sns.barplot(x=missing_data.index, y=missing_data.values, palette='viridis')
    plt.title('Count of Missing Values per Feature')
    plt.ylabel('Number of Missing Values')
    plt.xlabel('Features')
    for index, value in enumerate(missing_data.values):
        plt.text(index, value + 5, str(value), ha='center', fontweight='bold')
else:
    plt.text(0.5, 0.5, 'No Missing Values Found', ha='center', va='center')
    plt.title('Missing Values Analysis')

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'missing_values.png'), dpi=150)
plt.close()

# 3. Clean Missing Values
print("Cleaning missing values...")
# - Age: Impute missing Age with median of Pclass and Sex groups
df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))

# - Embarked: Impute missing with mode (most frequent value)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# - Cabin: Impute with 'Unknown' and create a helper feature HasCabin
df['HasCabin'] = df['Cabin'].notnull().astype(int)
df['Cabin'] = df['Cabin'].fillna('Unknown')

# 4. Outlier Analysis and Treatment (before & after boxplots)
print("Generating outlier boxplots...")
# We will focus on 'Fare' and 'Age' for outlier analysis
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(ax=axes[0], y=df['Age'], color='skyblue')
axes[0].set_title('Boxplot of Age (Before)')
sns.boxplot(ax=axes[1], y=df['Fare'], color='salmon')
axes[1].set_title('Boxplot of Fare (Before)')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'boxplot_before.png'), dpi=150)
plt.close()

# Outlier treatment using the IQR method for Fare (since it has extreme values)
Q1_fare = df['Fare'].quantile(0.25)
Q3_fare = df['Fare'].quantile(0.75)
IQR_fare = Q3_fare - Q1_fare
fare_lower = max(0, Q1_fare - 1.5 * IQR_fare)
fare_upper = Q3_fare + 1.5 * IQR_fare

print(f"Fare Outlier bounds: {fare_lower} to {fare_upper}")
df['Fare'] = df['Fare'].clip(lower=fare_lower, upper=fare_upper)

# Outlier treatment using the IQR method for Age
Q1_age = df['Age'].quantile(0.25)
Q3_age = df['Age'].quantile(0.75)
IQR_age = Q3_age - Q1_age
age_lower = max(0, Q1_age - 1.5 * IQR_age)
age_upper = Q3_age + 1.5 * IQR_age

print(f"Age Outlier bounds: {age_lower} to {age_upper}")
df['Age'] = df['Age'].clip(lower=age_lower, upper=age_upper)

# Boxplot after handling outliers
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(ax=axes[0], y=df['Age'], color='lightblue')
axes[0].set_title('Boxplot of Age (After Capping)')
sns.boxplot(ax=axes[1], y=df['Fare'], color='lightcoral')
axes[1].set_title('Boxplot of Fare (After Capping)')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'boxplot_after.png'), dpi=150)
plt.close()

# 5. Feature Engineering
print("Performing Feature Engineering...")
# - Extract Title from Name
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Normalize/clean titles
title_mapping = {
    'Mlle': 'Miss',
    'Ms': 'Miss',
    'Mme': 'Mrs',
    'Lady': 'Rare',
    'Countess': 'Rare',
    'Capt': 'Rare',
    'Col': 'Rare',
    'Don': 'Rare',
    'Dr': 'Rare',
    'Major': 'Rare',
    'Rev': 'Rare',
    'Sir': 'Rare',
    'Jonkheer': 'Rare',
    'Dona': 'Rare'
}
df['Title'] = df['Title'].replace(title_mapping)
# Keep common titles (Mr, Mrs, Miss, Master, Rare)
standard_titles = ['Mr', 'Mrs', 'Miss', 'Master', 'Rare']
df['Title'] = df['Title'].apply(lambda x: x if x in standard_titles else 'Mr')

# - FamilySize = SibSp + Parch + 1
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# - IsAlone = 1 if FamilySize == 1 else 0
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# 6. Feature Encoding
print("Encoding categorical variables...")
# One-hot encode Sex, Embarked, and Title
df_encoded = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Title'], drop_first=True, dtype=int)

# 7. Scale Numerical Features
print("Scaling numerical columns...")
scaler = StandardScaler()
numerical_cols = ['Age', 'Fare', 'FamilySize']
df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])

# 8. Drop unnecessary columns that won't be useful for models directly
# Keep a record of raw PassengerId, Survived, Name, Ticket, Cabin in case they are needed,
# or drop columns that are hard to feed into models like Name, Ticket, Cabin (since we have HasCabin and Title)
columns_to_drop = ['Name', 'Ticket', 'Cabin']
df_cleaned = df_encoded.drop(columns=columns_to_drop)

# Save the final cleaned data
print(f"Saving cleaned dataset to {CLEANED_DATA_PATH}...")
df_cleaned.to_csv(CLEANED_DATA_PATH, index=False)
print("Data preprocessing and cleaning completed successfully!")
