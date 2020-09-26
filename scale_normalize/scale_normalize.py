#%%
import pandas as pd
from path import Path
from sklearn.preprocessing import StandardScaler
import numpy as np


#%%
file_path = Path("scale_normalize_module/17-6-4-scale/Resources/loans_data_encoded.csv")
encoded_df = pd.read_csv(file_path)
encoded_df.head()

# %%
data_scaler = StandardScaler()


#%%
# Fit and transform methods combined together, but could use fit() then transform()
loans_data_scaled = data_scaler.fit_transform(encoded_df)
loans_data_scaled[:5]

# %%
# After rescaling, Mean and std:
print(np.mean(loans_data_scaled[:,0]))
print(np.std(loans_data_scaled[:,0]))

# %%
loans_data_scaled.shape

# %%
# For loop to check that all columns have been standardized
for i in range(0,loans_data_scaled.shape[1]):
    print(f"{i} column:")
    print(f"   mean:{np.mean(loans_data_scaled[:,i])}")
    print(f"   std:{np.std(loans_data_scaled[:,i])}")

# %%
