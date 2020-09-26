#%%
import pandas as pd
from path import Path

from sklearn.preprocessing import LabelEncoder


#%%
file_path = Path("encode_labels_module/17-6-1-label_encode/Resources/loans_data.csv")
loans_df = pd.read_csv(file_path)
loans_df.head()

# %%
# ENCODING (a preprocessing step)
# Scikit-learn’s machine learning algorithms, the text features 
# (month, education, and gender) will have to be converted 
# into numbers. This process is called encoding.

# WITH PANDAS :
# ***********************
# Encoding Gender (get_dummies method)
loans_binary_encoded = pd.get_dummies(loans_df, columns=["gender"])
loans_binary_encoded.head()

# %%
# Encoding multiple columns:
loans_binary_encoded = pd.get_dummies(loans_df, columns=["education", "gender"])
loans_binary_encoded.head()

# %%
# WITH SKLEARN:
# ***********************
# the encoding is performed in the same column with valuese 0-n instead of
# one column per value with 0-1 
le = LabelEncoder()
df2 = loans_df.copy()
df2['education'] = le.fit_transform(df2['education'])
df2['gender'] = le.fit_transform(df2['gender'])
df2.head()

# %%
# CUSTOM ENCODING:
# *************************
# For example to encode months we want Jan=1, Jul=7, Oct=10, etc...
# instead of random assignment that happens with LabelEncoder as we can see below
loans_df['month_le'] = le.fit_transform(loans_df['month'])
loans_df.head()

# %%
# Create custom dictionary
months_num = {
   "January": 1,
   "February": 2,
   "March": 3,
   "April": 4,
   "May": 5,
   "June": 6,
   "July": 7,
   "August": 8,
   "September": 9,
   "October": 10,
   "November": 11,
   "December": 12,
}
# Apply a lambda function
loans_df['month_num'] = loans_df['month'].apply(lambda x: months_num[x])
loans_df.head()
# %%
loans_df = loans_df.drop(["month", "month_le"], axis=1)
loans_df.head()

# %%
# Skill Drill: Encode the following labels of the dataset: month, education, and gender
# (month custom encoding, education and gender pandas encoding)
loans_encoded = loans_df.copy()
loans_encoded = pd.get_dummies(loans_encoded, columns=["education", "gender"])
loans_encoded.head()

# %%
loans_encoded.to_csv("./loans_data_encoded.csv")
# %%
