import arff
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.validation import check_is_fitted
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

# Ensure 'Amount' and 'Time' are numeric and non-null
df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
df.dropna(subset=['Amount', 'Time'], inplace=True)

# Normalize 'Amount' and 'Time'
df['normAmount'] = StandardScaler().fit_transform(df[['Amount']])
df['normTime'] = StandardScaler().fit_transform(df[['Time']])
df.drop(['Amount', 'Time'], axis=1, inplace=True)

# Split features and labels
X = df.drop('Class', axis=1)
y = df['Class']

# Handle class imbalance using SMOTE
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)
print("After resampling, counts of label '1': {}".format(sum(y_resampled == 1)))
print("After resampling, counts of label '0': {}".format(sum(y_resampled == 0)))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_test:", y_test.shape)

print("Unique classes in y_train:", set(y_train))
print("Class distribution in y_train:")
print(pd.Series(y_train).value_counts())

# Initialize RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train model with error handling
try:
    print("Starting model training...")
    model.fit(X_train, y_train)
    print("Model trained successfully.")
except Exception as e:
    print("Error during model training:", e)
    raise

# Check if model is fitted before predicting
try:
    check_is_fitted(model)
    print("Model is fitted, proceeding to prediction.")
except Exception as e:
    print("Model is not fitted:", e)
    raise

# Predict and evaluate
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

# Save the trained model
joblib.dump(model, 'model.pkl')
print("Model saved to model.pkl")
