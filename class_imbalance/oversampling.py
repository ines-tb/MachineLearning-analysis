#%%
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from collections import Counter
from sklearn.model_selection import train_test_split
# RANDOM
from imblearn.over_sampling import RandomOverSampler

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from imblearn.metrics import classification_report_imbalanced

# SYNTHETIC
from imblearn.over_sampling import SMOTE

#%%
# There are two techniques to deal with oversampling:
#  - Random Undersampling.
#  - Synthetic Minority Oversampling Technique (SMOTE).

#%%
# RANDOM OVERSAMPLING:
# **********************************
X, y = make_blobs(n_samples=[600, 60], random_state=1, cluster_std=5)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

# %%
# Split in train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
Counter(y_train) # => This counter confirms the imbalance

#%%
# Randomly oversample the minority class with the imblearn library
ros = RandomOverSampler(random_state=1)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
Counter(y_resampled)  # => This counter shows that resampled data is now balanced

# %%
# => Now, with a balanced dataset we can use a model to predict values 

# Logistic Regression model
model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(X_resampled, y_resampled)

# %%
# Predict and confusion matrix 
y_pred = model.predict(X_test)
confusion_matrix(y_test, y_pred)

# %%
# Accuracy
balanced_accuracy_score(y_test, y_pred) 
#  => Being 91% accurate can be misleading, so we need to 
#      check all possible elements with the classification_report_imbalanced

# %%
print(classification_report_imbalanced(y_test, y_pred))


#%%
# Synthetic Minority Oversampling Technique (SMOTE)
# *****************************************************
# Instead of random from the sample, it will create 'synthetic' values based on its distance from its neighbors.
# And so, a deficiency of SMOTE is its vulnerability to outliers as synthetic values can be created from outliers. 

# Sampling_strategy = auto will by default resample the minority to the same count of the majority
X_resampled, y_resampled = SMOTE(random_state=1,
sampling_strategy='auto').fit_resample(
   X_train, y_train)
Counter(y_resampled)

# %%
# Now lets train and predict with Logistic Regression
model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(X_resampled, y_resampled)

y_pred = model.predict(X_test)
balanced_accuracy_score(y_test, y_pred)

confusion_matrix(y_test, y_pred)

print(classification_report_imbalanced(y_test, y_pred))

# %%
