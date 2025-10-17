import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv("students.csv")

# Split features and label
X = df.drop('passed', axis=1)
y = df['passed']

# Identify column types
num_cols = ['age', 'study_hours']
cat_cols = ['gender', 'school']

# Create transformers
num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine both
preprocessor = ColumnTransformer([
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])

# Add model to pipeline
clf = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression())
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit & predict
clf.fit(X_train, y_train)
print("Model accuracy:", clf.score(X_test, y_test))
