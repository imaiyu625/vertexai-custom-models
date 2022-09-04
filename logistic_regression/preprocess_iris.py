import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
df_iris['class'] = iris.target

df_iris = df_iris[df_iris['class'] != 2]

y_ = df_iris['class'].to_numpy().reshape(df_iris['class'].shape[0], 1)
x_ = df_iris[iris.feature_names].to_numpy()

pd.DataFrame(np.concatenate((y_, x_), axis=1)).to_csv('data.csv', header=None, index=None)
