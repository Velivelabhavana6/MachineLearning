import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv('Missing_data.csv')

# Separate features and labels
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Handle missing data if any
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])  # assuming 1:3 are numeric columns with missing values

# Encode categorical column (assuming 'Country' is at index 0)
ct = ColumnTransformer(transformers=[
    ('encoder', OneHotEncoder(), [0])
], remainder='passthrough')
X = ct.fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Apply Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Output to verify
# print(X_test)
print(X_train)