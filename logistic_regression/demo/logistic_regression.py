#%%
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

#%%
# SUMMARY OF LOGISTIC REGRESSION MODEL:
# ---------------------------------------------
# 1. Create a model with LogisticRegression().
# 2. Train the model with model.fit().
# 3. Make predictions with model.predict().
# 4. Validate the model with accuracy_score(). 

#%%
X, y = make_blobs(centers=2, random_state=42)

print(f"Labels: {y[:10]}")
print(f"Data: {X[:10]}")
# %%
plt.scatter(X[:, 0], X[:, 1], c=y)

# %%
# Training and testing the model
X_train, X_test, y_train, y_test = train_test_split(X,
    y, random_state=1, stratify=y)
# %%
# Instantiate a Logistic Regression Model

classifier = LogisticRegression(solver='lbfgs', random_state=1)
classifier

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
   intercept_scaling=1, l1_ratio=None, max_iter=100,
   multi_class='warn', n_jobs=None, penalty='12',
   random_state=1, solver='lbfgs', tol=0.0001, verbose=0,
   warm_start=False)
# %%
# Train the Logistic Regression model
classifier.fit(X_train, y_train)

#%%
# Validate the logistic regression model
predictions = classifier.predict(X_test)
pd.DataFrame({"Prediction": predictions, "Actual": y_test})
# Evaluate the performance of the predictions
accuracy_score(y_test, predictions)

#%%
# Classify if the next point is purple or yellow
new_data = np.array([[-2, 6]])
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.scatter(new_data[0, 0], new_data[0, 1], c="r", marker="o", s=100)
plt.show()
# %%
predictions = classifier.predict(new_data)
print("Classes are either 0 (purple) or 1 (yellow)")
print(f"The new point was classified as: {predictions}")
# %%
