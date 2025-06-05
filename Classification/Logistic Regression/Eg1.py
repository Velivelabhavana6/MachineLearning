import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score
# Load dataset
df = pd.read_csv('Balanced.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train Logistic Regression
classifier = LogisticRegression(random_state=42)
classifier.fit(X_train, y_train)

# Predict a new result (e.g., Age=27, Salary=86557)
new_data = sc.transform([[21, 44800]])
res = classifier.predict(new_data)
# print(res)

#predicting a test set res
y_pred=classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

#making the confusion matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)
ac=accuracy_score(y_test,y_pred)
print(ac)

#Visualizing the training set res
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Inverse transform the scaled features to plot correctly
X_set, y_set = sc.inverse_transform(X_train), y_train

# Create the meshgrid for plotting decision boundary
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.25),
    np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=250)
)

# Plot the decision boundary using contourf
plt.contourf(
    X1, X2,
    classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
    alpha=0.75,
    cmap=ListedColormap(('red', 'green'))
)

# Plotting limits
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# Scatter plot of actual data points
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
        X_set[y_set == j, 0],
        X_set[y_set == j, 1],
        c=ListedColormap(('red', 'green'))(i),
        label=j
    )

# Labels and title
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
# plt.legend()
# plt.show()



#VISUALIZING THE TEST RESULT
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Assuming X_test and y_test are already defined and scaled using StandardScaler
# Also assuming 'classifier' is the trained Logistic Regression model
# And 'sc' is the StandardScaler used to scale the features

# Inverse transform the scaled test data
X_set, y_set = sc.inverse_transform(X_test), y_test

# Creating a meshgrid for plotting decision boundaries
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.25),
    np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.25)
)

# Plotting the decision boundary
plt.contourf(
    X1, X2,
    classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
    alpha=0.75,
    cmap=ListedColormap(('red', 'green'))
)

# Plot limits
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# Plotting the actual test points
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
        X_set[y_set == j, 0],
        X_set[y_set == j, 1],
        c=ListedColormap(('red', 'green'))(i),
        label=j
    )

plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()