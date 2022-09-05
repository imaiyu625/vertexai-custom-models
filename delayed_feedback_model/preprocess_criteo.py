import pandas as pd
import numpy as np

header = ['ts_click', 'ts_cv', 'int1', 'int2', 'int3', 'int4', 'int5', 'int6', 'int7', 'int8', 'cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8', 'cat9']
df_criteo = pd.read_table('data.txt', names=header)

# remove rows where ts_click is greater than ts_cv
df_criteo.drop(df_criteo.index[df_criteo.ts_click > df_criteo.ts_cv], axis=0, inplace=True)
df_criteo = df_criteo.sort_values('ts_click').reset_index(drop=True)

# set temporary current time
ts_current = 20000
df_criteo = df_criteo[df_criteo.ts_click <= ts_current]
df_criteo['ob_cv'] = ~np.isnan(df_criteo.ts_cv) * 1
df_criteo.ob_cv[df_criteo.ts_cv > ts_current] = 0

# for observed cv users, calculate the time difference between ts_cv and ts_click
# for unobserved users, fill in the elapsed time from ts_click
df_criteo['ts_elapsed'] = df_criteo.ts_cv.fillna(ts_current) - df_criteo.ts_click

# normalize timestamps, use t=0.5 if predict the cv within the next 10000 elapsed times
df_criteo.ts_elapsed /= ts_current

# bucketizing the integer features on a logarithmic scale
for feature in ['int1', 'int2', 'int3', 'int4', 'int5', 'int6', 'int7', 'int8']:
    df_criteo[feature][np.isnan(df_criteo[feature])] = 0
    col = df_criteo[feature].to_numpy()
    col = np.where(col > 0, (np.log(col + 0.5) / np.log(1.5)).astype(int), col)
    df_criteo[feature] = col.astype(int)

df_criteo[['ob_cv', 'ts_elapsed', 'int1', 'int2', 'int3', 'int4', 'int5', 'int6', 'int7', 'int8']].to_csv('data.csv', header=None, index=None)
