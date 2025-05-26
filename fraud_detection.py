import arff
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load ARFF file
with open('creditcard.arff', 'r') as f:
    data = arff.load(f)

# Convert to DataFrame
df = pd.DataFrame(data['data'], columns=[attr[0] for attr in data['attributes']])
print(df.head())
print("Dataset Loaded. Shape:", df.shape)

# Ensure 'Class' column is numeric
df['Class'] = pd.to_numeric(df['Class'], errors='coerce')
df.dropna(subset=['Class'], inplace=True)
df['Class'] = df['Class'].astype(int)

# Check class distribution
print("Class distribution:\n", df['Class'].value_counts())

# Preprocessing: normalize 'Amount' and 'Time'
df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
df.dropna(subset=['Amount', 'Time'], inplace=True)

df['normAmount'] = StandardScaler().fit_transform(df[['Amount']])
df['normTime'] = StandardScaler().fit_transform(df[['Time']])
df.drop(['Amount', 'Time'], axis=1, inplace=True)

# Split features and label
X = df.drop('Class', axis=1)
y = df['Class']

# Handle imbalance using SMOTE
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)
print("After resampling, counts of label '1': {}".format(sum(y_resampled == 1)))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# Train RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model trained.")

# Predict & Evaluate
y_pred = model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))

# Visualization
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.show()

# Save model
joblib.dump(model, 'model.pkl')
print("Model saved to model.pkl")

