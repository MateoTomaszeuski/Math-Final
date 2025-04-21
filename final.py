# %%
import os
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve

# Ensure Kaggle credentials are configured:
# Place your kaggle.json in ~/.kaggle/kaggle.json with permissions 600.
# Step By Step (MAC)
# Sign in at https://www.kaggle.com → My Account → “Create New API Token.”
# mkdir -p ~/.kaggle
# mv ~/Downloads/kaggle.json ~/.kaggle/
# chmod 600 ~/.kaggle/kaggle.json

# %%
# Download and unzip dataset via Kaggle API
dataset = 'thedevastator/higher-education-predictors-of-student-retention'
data_dir = 'data'
os.makedirs(data_dir, exist_ok=True)
api = KaggleApi()
api.authenticate()
api.dataset_download_files(dataset, path=data_dir, unzip=True)

# %%
# Load CSV
csv_filename = 'dataset.csv'
csv_path = os.path.join(data_dir, csv_filename)
df = pd.read_csv(csv_path)
print("Columns in dataset:", df.columns.tolist())

# %%
# Data Cleaning & Encoding
df.drop_duplicates(inplace=True)
label_map = {'Dropout': 0, 'Enrolled': 1, 'Graduate': 2}
df['Target'] = df['Target'].map(label_map)
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
if 'Target' in cat_cols:
    cat_cols.remove('Target')
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# %%
# Train/Test Split & Scaling
X = df.drop('Target', axis=1)
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# %%
# Exploratory Visualizations

# Age at Enrollment Distribution
# Purpose: To see how student ages are spread at the time they enroll.
# What it shows:
# The shape of the age distribution (e.g. bell‑shaped, skewed, multimodal)
# Common age ranges (where the bars are tallest)
# Any outliers or unusual age brackets (e.g. very young or older students)
plt.figure()
plt.hist(df['Age at enrollment'].dropna(), bins=20)
plt.title('Age at Enrollment Distribution')
plt.xlabel('Age at Enrollment')
plt.ylabel('Count')
plt.show()

# %%
# Student Outcome Distribution

# Purpose: To compare the raw counts of each outcome class—Dropout, Enrolled, Graduate.
# What it shows:
# Which outcome is most common in the dataset
# Whether classes are balanced or skewed (e.g., far more Enrolled cases than Dropouts)
plt.figure()
outcomes = df['Target'].map({0:'Dropout',1:'Enrolled',2:'Graduate'})
counts = outcomes.value_counts()
plt.bar(counts.index, counts.values)
plt.title('Student Outcome Distribution')
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.show()

# %%
# Correlation Heatmap

# Purpose: To quantify the pairwise linear relationships between every pair of numeric features (and the encoded target).
# What it shows:
# Positive correlations (values close to +1) and negative correlations (values close to –1) via color intensity
# Which features tend to move together (e.g. “Curricular units enrolled” vs. “credited”)
plt.figure(figsize=(10,8))
corr = df.select_dtypes(include=[np.number]).corr()
plt.imshow(corr, aspect='auto')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=90)
plt.yticks(range(len(corr)), corr.columns)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# %%
