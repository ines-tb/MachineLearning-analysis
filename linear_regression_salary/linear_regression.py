#%%
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# %%
df = pd.read_csv(Path('.\\Resources\\Salary_Data.csv'))
df.head()
# %%
# Visually inspect the relationship between Years of Experience and Salary
plt.scatter(df.YearsExperience,df.Salary)
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
# %%
# Format the data to meet the requirements of the Scikit-learn library
X = df.YearsExperience.values.reshape(-1, 1)
X[:5]
# %%
# Shape of X
X.shape
# %%
y = df.Salary
# %%
# Instance of the linear regression model
model = LinearRegression()
# %%
# Learning Stage (fitting or training)
model.fit(X,y)

#%%
# Generate predictions 
yPred = model.predict(X)
print(yPred.shape)

#%%
# Plot predictions 
plt.scatter(X,y)
plt.plot(X,yPred,color='red')
plt.show()
# %%
# Calculate slope (model.coef_) and y-intercept (model.intercept_) of the model
print(model.coef_)
print(model.intercept_)

# %%
