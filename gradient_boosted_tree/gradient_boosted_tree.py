#%%
import pandas as pd
from path import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#%%
file_path = Path("gradient_boosted_tree_module/17-9-3-gradient_boosted_tree/Resources/loans_data_encoded.csv")
loans_df = pd.read_csv(file_path)
loans_df.head()

# %%
# Separate feature from target
X = loans_df.copy()
X = X.drop("bad", axis=1)
y = loans_df["bad"].values

# %%
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,
   y, random_state=1)

# %%
# Scale data 
# Not strictly necessary but good when comparing models
scaler = StandardScaler()
X_scaler = scaler.fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

# %%
# Identify the learning rate that yields the best performance.
# Create a classifier object
learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
for learning_rate in learning_rates:
    classifier = GradientBoostingClassifier(n_estimators=20,
                                            learning_rate=learning_rate,
                                            max_features=5,
                                            max_depth=3,
                                            random_state=0)

    # Fit the model
    classifier.fit(X_train_scaled, y_train)
    print("Learning rate: ", learning_rate)

    # Score the model
    print("Accuracy score (training): {0:.3f}".format(
        classifier.score(
            X_train_scaled,
            y_train)))
    print("Accuracy score (validation): {0:.3f}".format(
        classifier.score(
            X_test_scaled,
            y_test)))
    print()
# => This for loop gives us the data for which the GradientBoostingClassifier model
#       performs well, in this case learning_rate=0.5, as some others performs well on 
#       training data but poorly on testing (Overfitting)

# %%
# Using the learning_rate value obtained from the for loop, 
#   we instantiate a model, train it, then create predictions.
classifier = GradientBoostingClassifier(n_estimators=20,
   learning_rate=0.5, max_features=5, max_depth=3, random_state=0)

classifier.fit(X_train_scaled, y_train)
predictions = classifier.predict(X_test_scaled)

# %%
# Assess model's performance
#   => Returns same score as classifier.score
acc_score = accuracy_score(y_test, predictions)
print(f"Accuracy Score : {acc_score}")

# %%
# Confusion matrix
cm = confusion_matrix(y_test, predictions)
cm_df = pd.DataFrame(
   cm, index=["Actual 0", "Actual 1"],
   columns=["Predicted 0", "Predicted 1"]
)
display(cm_df)

# %%
# Clasification report
print("Classification Report")
print(classification_report(y_test, predictions))
# %%
