#%%
# Initial imports
import pandas as pd
from path import Path
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# %%
# Loading data
file_path = Path("decision_tree_module/17-7-2-decision_tree/Resources/loans_data_encoded.csv")
df_loans = pd.read_csv(file_path)
df_loans.head()

# %%
# Define the features set.
X = df_loans.copy()
X = X.drop("bad", axis=1)
X.head()

# %%
# Define the target set.
y = df_loans["bad"].values
y[:5]

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=78)

# %%
# Determine the shape of our training and testing sets.
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# %%
# Splitting into Train and Test sets into an 80/20 split.
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, random_state=78, train_size=0.80)

# %%
# Determine the shape of our training and testing sets.
print(X_train2.shape)
print(X_test2.shape)
print(y_train2.shape)
print(y_test2.shape)

# %%
# Decision trees do not required scaled data (mean=0, std=1) but it is useful for comparing models
# Creating a StandardScaler instance.
scaler = StandardScaler()
# Fitting the Standard Scaler with the training data.
X_scaler = scaler.fit(X_train)

# Scaling the data.
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

# %%
# Creating the decision tree classifier instance.
model = tree.DecisionTreeClassifier()
# Fitting the model.
model = model.fit(X_train_scaled, y_train)

# %%
# Making predictions using the testing data.
predictions = model.predict(X_test_scaled)
predictions

# %%
# EVALUATE THE MODEL:
# *************************************
# Calculating the confusion matrix
cm = confusion_matrix(y_test, predictions)

# Create a DataFrame from the confusion matrix.
cm_df = pd.DataFrame(
    cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])

cm_df

# %%
# Calculating the accuracy score.
# It can also be calculated by:
#   (True Positives (TP) + True Negatives (TN)) / Total 
#   = (52 + 20)/125 = 0.576
acc_score = accuracy_score(y_test, predictions)
acc_score

# %%
# EXPLANATION OF THE SUMMAR IN SECTION 17.7.3

# Displaying results
print("Confusion Matrix")
display(cm_df)
print(f"Accuracy Score : {acc_score}")
print("Classification Report")
print(classification_report(y_test, predictions))
# %%
