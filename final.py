# %%
import os
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.discrete.discrete_model import Probit, Logit
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Our Question: Are factors like gender, age at enrollment,
# parent occupation, financial aid, and grades 
# related to student dropout rates?

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
print(df.info())

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
plt.figure()
plt.hist(df['Age at enrollment'].dropna(), bins=20)
plt.title('Age at Enrollment Distribution')
plt.xlabel('Age at Enrollment')
plt.ylabel('Count')
plt.show()

# %%
# Student Outcome Distribution
plt.figure()
outcomes = df['Target'].map({0:'Dropout',1:'Enrolled',2:'Graduate'})
counts = outcomes.value_counts()
plt.bar(counts.index, counts.values)
plt.title('Student Outcome Distribution')
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.show()

# %%
dropout = df[df['Target'] == 0]  
plt.figure()
plt.bar(dropout['Gender'].value_counts().index, dropout['Gender'].value_counts().values)
plt.title('Gender Distribution Among Dropouts')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.xticks([0, 1],['Female', 'Male'])  # Not sure which is male and female, datased not clear.
plt.show()

# %%
df['Weighted_Avg_Grade'] = (
    (df['Curricular units 1st sem (grade)'] * df['Curricular units 1st sem (enrolled)']) +
    (df['Curricular units 2nd sem (grade)'] * df['Curricular units 2nd sem (enrolled)'])
) / (
    df['Curricular units 1st sem (enrolled)'] + df['Curricular units 2nd sem (enrolled)']
).dropna()

plt.figure()
sns.scatterplot(
    x=(df['Curricular units 1st sem (enrolled)'] + df['Curricular units 2nd sem (enrolled)']),
    y=df['Weighted_Avg_Grade'],
    hue=outcomes
)
plt.title('Number of Enrolled Curricular Units vs Weighted Average Grade')
plt.xlabel('Total Curricular Units Enrolled')
plt.ylabel('Weighted Average Grade')
plt.legend(title='Outcome')
plt.show()

# %%
# Ordinary Least Squares (OLS) Model
df['Total Credits'] = df['Curricular units 1st sem (enrolled)'] + df['Curricular units 2nd sem (enrolled)']
df.dropna(subset=[
    'Weighted_Avg_Grade', 
    'Target'
], inplace=True)

X = df[['Age at enrollment',
        'Weighted_Avg_Grade',
        'Gender',
        'Scholarship holder',
        'Debtor',
        'Tuition fees up to date'
        ]]
y = df['Target']

X = sm.add_constant(X)

ols_model = sm.OLS(y, X).fit()

print(ols_model.summary())
# %%
# Probit and Logit Models

# Convert dropout to a boolean (1 for Dropout, 0 otherwise)
df['Dropout_Boolean'] = (df['Target'] == 0).astype(int)
y = df['Dropout_Boolean']

# Fit the Probit model
probit_model = Probit(y, X).fit()
print("Probit Model Summary:")
print(probit_model.summary())
print("Probit Model AIC:")
print(probit_model.aic)
#%%
# Fit the Logit model
logit_model = Logit(y, X).fit()
print("Logit Model Summary:")
print(logit_model.summary())
print("Logit Model AIC:")
print(logit_model.aic)

# %%
# Predict probabilities
y_pred_prob = logit_model.predict(X)
# Threshold at 0.5 to get predicted classes
y_pred_class = (y_pred_prob >= 0.5).astype(int)

# Create confusion matrix
cm = confusion_matrix(y, y_pred_class)

# Plot heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Dropout', 'Dropout'],
            yticklabels=['Not Dropout', 'Dropout'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.show()
# %%
