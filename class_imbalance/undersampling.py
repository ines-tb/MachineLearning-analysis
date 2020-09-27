#%%
import pandas as pd
from path import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from imblearn.metrics import classification_report_imbalanced

# Cluster Centroid
from imblearn.under_sampling import ClusterCentroids

# Synthetic Oversampling
from imblearn.over_sampling import SMOTE


#%%
# There are two techniques to deal with oversampling:
#  - Random Undersampling.
#  - Synthetic Minority Oversampling Technique (SMOTE).



#%%
# Random Undersampling
# **************************
data = Path('undersampling_module/17-10-2-undersampling/Resources/cc_default.csv')
df = pd.read_csv(data)
df.head()

# %%
# CSV LEGEND:
#   ln_balance_limit: maximum balance limit on a card
#   sex: 1 = female, 0 = sex
#   education: 1 = graduate school, 2 = university, 3 = high school, 4 = others
#   marriage: 1 = married, 0 = single
#   age: age of credit card holder
#   default_next_month: 1 = yes, 0 = no

#%%
x_cols = [i for i in df.columns if i not in ('ID', 'default_next_month')]
X = df[x_cols]
y = df['default_next_month']

# %%
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1)

# %%
ros = RandomUnderSampler(random_state=1)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
Counter(y_resampled)

#%%
model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(X_resampled, y_resampled)

#%%
# Predict and Generate the metrics 
y_pred = model.predict(X_test)
confusion_matrix(y_test, y_pred)

balanced_accuracy_score(y_test, y_pred)

print(classification_report_imbalanced(y_test, y_pred))


#%%
# Cluster Centroid Undersampling
# ******************************************
# Instantiate the resampling module:
cc = ClusterCentroids(random_state=1)
X_resampled, y_resampled = cc.fit_resample(X_train, y_train)

#%%
# Instantiate and train a logistic regression model
model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(X_resampled, y_resampled)

#%%
# Predict and Generate the metrics
y_pred = model.predict(X_test)
confusion_matrix(y_test, y_pred)

balanced_accuracy_score(y_test, y_pred)

print(classification_report_imbalanced(y_test, y_pred))

#%%
# COMPARISON OVERSAMPLING METHOD VS UNDERSAMPLING

X_overResampled, y_overResampled = SMOTE(random_state=1,
sampling_strategy='auto').fit_resample(
   X_train, y_train)
Counter(y_overResampled)

# %%
# Now lets train and predict with Logistic Regression
model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(X_overResampled, y_overResampled)

y_OverPred = model.predict(X_test)
balanced_accuracy_score(y_test, y_OverPred)

confusion_matrix(y_test, y_OverPred)

print(classification_report_imbalanced(y_test, y_OverPred))
# %%
