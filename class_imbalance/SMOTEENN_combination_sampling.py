#%%
import pandas as pd
from path import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from sklearn.metrics import balanced_accuracy_score
from imblearn.metrics import classification_report_imbalanced


#%%
# STEPS OF SMOTEENN:
# 1. Oversample the minority class with SMOTE.
# 2. Clean the resulting data with an undersampling strategy. 
#       If the two nearest neighbors of a data point belong to two 
#       different classes, that data point is dropped.

#%%
data = Path('combination_sampling_module/17-10-3-combination_sampling/Resources/cc_default.csv')
df = pd.read_csv(data)
df.head()

# %%
# Create feaures and target dataset
x_cols = [i for i in df.columns if i not in ('ID', 'default_next_month')]
X = df[x_cols]
y = df['default_next_month']

# %%
# Split in training and testing 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

#%%
# Create instance of SMOTEENN
smote_enn = SMOTEENN(random_state=0)
X_resampled, y_resampled = smote_enn.fit_resample(X, y)

#%%
# Logistic Regression model
model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(X_resampled, y_resampled)


#%%
# Prediction and metrics
y_pred = model.predict(X_test)
confusion_matrix(y_test, y_pred)

balanced_accuracy_score(y_test, y_pred)

print(classification_report_imbalanced(y_test, y_pred))