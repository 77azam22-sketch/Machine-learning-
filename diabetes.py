import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# 1. Load Dataset
df = pd.read_csv('C:/Users/USER PC/Music/diabetes_prediction_dataset.csv')

# 2. Preprocessing (Convert text to numbers)
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
df['smoking_history'] = le.fit_transform(df['smoking_history'])

# 3. Define Features (X) and Target (y)
X = df.drop('diabetes', axis=1)
y = df['diabetes']

# 4. Split into Train/Test (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# 5. Train Decision Tree
dt = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
dt.fit(X_train, y_train)

# 6. Predict & Evaluate
y_pred = dt.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Plot Decision Tree
plt.figure(figsize=(20,10))
plot_tree(dt, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'], filled=True, fontsize=12)
plt.title("Decision Tree for Diabetes Prediction")
plt.savefig('decision_tree_diabetes.png')