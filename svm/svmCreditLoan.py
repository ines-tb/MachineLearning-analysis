#%%
from path import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression

#%%
data = Path('svm_module/17-5-2-svm/Resources/loans.csv')
df = pd.read_csv(data)
df.head()

# %%
y = df["status"]
X = df.drop(columns="status")

# %%
X_train, X_test, y_train, y_test = train_test_split(X,
   y,  random_state=1, stratify=y)
X_train.shape

# %%
# Instantiate the model
model = SVC(kernel='linear')

#%%
# Train the model
model.fit(X_train,y_train)

#%%
# Predict with the model
y_pred = model.predict(X_test)
results = pd.DataFrame({
   "Prediction": y_pred,
   "Actual": y_test
}).reset_index(drop=True)
results.head()

# %%
# Assess accuracy
accuracy_score(y_test, y_pred)

# %%
# Confusion matrix and clasification report
matrix = confusion_matrix(y_test, y_pred)
print(matrix)

#%%
print(classification_report(y_test, y_pred))

# %%
#*****************************************************
# Compare this svm model with logistic
# Training and testing the model
X_train_LogReg, X_test_LogReg, y_train_LogReg, y_test_LogReg = train_test_split(X,
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
classifier.fit(X_train_LogReg, y_train_LogReg)

#%%
# Validate the logistic regression model
predictions = classifier.predict(X_test_LogReg)
pd.DataFrame({"Prediction": predictions, "Actual": y_test_LogReg})
# Evaluate the performance of the predictions
accuracy_score(y_test_LogReg, predictions)


# %%
# Confusion matrix and clasification report
matrix = confusion_matrix(y_test_LogReg, predictions)
print(matrix)

#%%
print(classification_report(y_test_LogReg, predictions))


# FOR APPROVAL SEEMS TO BE BETTER THE svm MODEL