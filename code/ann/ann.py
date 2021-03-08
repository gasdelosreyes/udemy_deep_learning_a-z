import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#Read the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,-1].values

# Label Encoder converts each unique value in a certain numerical value.
# Recommended for values that have two values like True or False (Example: Female or Male)
# It's for better performance
le = LabelEncoder()

X[:,2] = le.fit_transform(X[:,2])

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])],remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)