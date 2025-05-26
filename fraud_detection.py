# credit_card_fraud_detection.py

import pandas as pd
import arff  # for reading ARFF format
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt


# STEP 2: Load the Dataset (ARFF format)
with open('creditcard.csv', 'r') as f:
    data = arff.load(f)

df = pd.DataFrame(data['data'], columns=[attr[0] for attr in data['attributes']])

print(df.head())
print("Dataset Loaded. Shape:", df.shape)

# STEP 3: Explore the Dataset
print(df.info())
print(df.describe())

# Fix column name for target if lowercase 'class' instead of 'Class'
if 'Class' not in df.columns and 'class' in df.columns:
    df.rename(columns={'class': 'Class'}, inplace=True)

print("Class distribution:\n", df['Class'].value_counts())

# STEP 4: Preprocess the Data
df['normAmount'] = StandardScaler().fit_transform(df[['Amount']])
df['normTime'] = StandardScaler().fit_transform(df[['Time']])
df.drop(['Amount', 'Time'], axis=1, inplace=True)

# STEP 5: Split Features and Labels
X = df.drop('Class', axis=1)
y = df['Class']

# STEP 6: Handle Imbalanced Dataset (Using SMOTE)
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)
print("After resampling, counts of label '1': {}".format(sum(y_resampled == 1)))

# STEP 7: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# STEP 8: Train a RandomForest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model trained.")

# STEP 9: Evaluate the Model
y_pred = model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))

# STEP 10: Visualization (Confusion Matrix)
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.show()
