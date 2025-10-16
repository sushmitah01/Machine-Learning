# =====================================
# 🧩 STEP 1: IMPORT LIBRARIES
# =====================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# (You can add specific models later, e.g.)
from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LinearRegression  # for regression projects


# =====================================
# 🧠 STEP 2: LOAD & EXPLORE DATA
# =====================================

# Example: Replace with your own CSV file
df = pd.read_csv("your_dataset.csv")

print("🔹 First 5 rows:")
print(df.head())

print("\n🔹 Data Info:")
print(df.info())

print("\n🔹 Missing Values:")
print(df.isnull().sum())

print("\n🔹 Basic Statistics:")
print(df.describe())


# =====================================
# 🔍 STEP 3: DATA CLEANING
# =====================================

# Drop unnecessary columns
# df.drop(['ID', 'Unnamed: 0'], axis=1, inplace=True)

# Handle missing values
# df['column_name'].fillna(df['column_name'].mean(), inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)


# =====================================
# 📊 STEP 4: EXPLORATORY DATA ANALYSIS (EDA)
# =====================================

# Example plots (customize for your data)
sns.countplot(x='target_column', data=df)
plt.title('Target Distribution')
plt.show()

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()


# =====================================
# ⚙️ STEP 5: SPLIT FEATURES AND TARGET
# =====================================

X = df.drop('target_column', axis=1)
y = df['target_column']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")


# =====================================
# 🧹 STEP 6: PREPROCESSING PIPELINE
# =====================================

# Identify column types
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# Numeric transformer
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Categorical transformer
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine both
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])


# =====================================
# 🤖 STEP 7: MODEL BUILDING
# =====================================

# Example: Random Forest for classification
model = RandomForestClassifier(random_state=42)

# Combine preprocessing + model
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# Train model
clf.fit(X_train, y_train)


# =====================================
# 🧪 STEP 8: EVALUATION
# =====================================

y_pred = clf.predict(X_test)

print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()


# =====================================
# 🔧 STEP 9: HYPERPARAMETER TUNING (Optional)
# =====================================

# param_grid = {
#     'model__n_estimators': [50, 100, 200],
#     'model__max_depth': [None, 5, 10]
# }

# grid = GridSearchCV(clf, param_grid, cv=3, scoring='accuracy')
# grid.fit(X_train, y_train)
# print("Best Params:", grid.best_params_)
# print("Best Score:", grid.best_score_)


# =====================================
# 🚀 STEP 10: SAVE MODEL (Optional)
# =====================================

# import joblib
# joblib.dump(clf, 'final_model.pkl')
# print("✅ Model saved successfully!")
