#%%
# Initial imports.
import pandas as pd
from path import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# %%
# Loading data
file_path = Path("decision_tree_module/17-7-2-decision_tree/Resources/loans_data_encoded.csv")
df_loans = pd.read_csv(file_path)
df_loans.head()

# %%
# PREPROCESSING STEPS before fiting the random forest model:
# 1. Define features and target
# 2. Split into training and testing sets
# 3. Create a StandardScaler instance
# 4. Fit the StandardScaler.
# 5. Scale the data

#%%
# Define the features set.
X = df_loans.copy()
#X = X.drop("bad", axis=1)
X = X.drop(["bad","gender_male","gender_female","education_Bachelor","education_High School or Below", "education_Master or Above","education_college"], axis=1)
X.head()

# %%
# Define the target set.
y = df_loans["bad"].ravel()
y[:5]

# %%
# Splitting into Train and Test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=78)

# %%
# Creating a StandardScaler instance.
scaler = StandardScaler()
# Fitting the Standard Scaler with the training data.
X_scaler = scaler.fit(X_train)

# Scaling the data.
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

# %%
# FIT THE RANDOM FOREST MODEL
# Create a random forest classifier.
# best practice is to use between 64 and 128 random forests
#rf_model = RandomForestClassifier(n_estimators=128, random_state=78) 
rf_model = RandomForestClassifier(n_estimators=500, random_state=78) 

# %%
# Fitting the model
rf_model = rf_model.fit(X_train_scaled, y_train)

# %%
# MAKE PREDICTIONS USING TESTING DATA:
# Making predictions using the testing data.
predictions = rf_model.predict(X_test_scaled)

# %%
# EVALUATE THE MODEL:
# Calculating the confusion matrix.
cm = confusion_matrix(y_test, predictions)

# Create a DataFrame from the confusion matrix.
cm_df = pd.DataFrame(
    cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])

cm_df

# %%
# Calculating the accuracy score.
acc_score = accuracy_score(y_test, predictions)

# %%
# Displaying results
print("Confusion Matrix")
display(cm_df)
print(f"Accuracy Score : {acc_score}")
print("Classification Report")
print(classification_report(y_test, predictions))

# %%
# RANK THE IMPORTANCE OF FEATURES:
# Allows us to see which features have the most impact on the decision.

# Calculate feature importance in the Random Forest model.
importances = rf_model.feature_importances_
importances

# %%
# We can sort the features by their importance.
sorted(zip(rf_model.feature_importances_, X.columns), reverse=True)
# To improve this model, we can drop some of the lower ranked features

# %%
