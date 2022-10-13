import os
import json
import pickle
import tempfile
from google.cloud import storage
from urllib.parse import urlparse
from io import BytesIO
import argparse
import pandas as pd
import numpy as np
import scipy.optimize as optimize
from sklearn.model_selection import KFold

# set env
data = os.environ['data']
model_dir = os.environ['model_dir']
random_state = int(os.environ['random_state'])
n_splits = int(os.environ['BATCH_TASK_COUNT'])
iter = int(os.environ['BATCH_TASK_INDEX'])

storage_client = storage.Client()

parsed_url = urlparse(data)
bucket = storage_client.bucket(parsed_url.netloc)
blob = bucket.blob(parsed_url.path.lstrip('/'))

def convert_data(df):
    y_ = df[0].to_numpy()
    x_ = df.drop(0, axis=1).to_numpy()
    x_ = np.concatenate((np.ones((x_.shape[0], 1)), x_), axis=1) # add bias term
    return x_, y_

def loss_func(w_, x_, y_):
    p_ = 1 / (1 + np.exp(-np.dot(w_, x_.T)))
    ll = np.dot(y_.T, np.log(p_)) + np.dot(1 - y_.T, np.log(1 - p_))
    return - ll / len(y_)

def fit(x_, y_):
    w_ = np.zeros(x_.shape[1])
    print('negative log likelihood:', loss_func(w_, x_, y_))
    wopt_ = optimize.minimize(loss_func, w_, args=(x_, y_), method='L-BFGS-B').x
    print('negative log likelihood:', loss_func(wopt_, x_, y_))
    return wopt_

# k-fold cross validation
raw_data = pd.read_csv(BytesIO(blob.download_as_bytes()), header=None)
kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
for i, (tr_indices, te_indices) in enumerate(kf.split(raw_data)):
    if i == iter:
        tr_data = raw_data.iloc[tr_indices]
        te_data = raw_data.iloc[te_indices]
        break

xtr_, ytr_ = convert_data(tr_data)
param = fit(xtr_, ytr_)

# model file
artifact_filename = 'model.pkl'
prefix = os.path.join(model_dir, str(iter))
storage_path = os.path.join(prefix, artifact_filename)

with tempfile.NamedTemporaryFile(mode='wb') as temp:
    pickle.dump(param, temp)
    temp.flush()
    blob = storage.blob.Blob.from_string(storage_path, client=storage_client)
    blob.upload_from_filename(temp.name)

# meta info
artifact_filename = 'meta.json'
storage_path = os.path.join(prefix, artifact_filename)

xte_, yte_ = convert_data(te_data)
with tempfile.NamedTemporaryFile(mode='w') as temp:
    json.dump({
        'tr_indices': tr_indices.tolist(),
        'te_indices': te_indices.tolist(),
        'tr_loss': loss_func(param, xtr_, ytr_),
        'te_loss': loss_func(param, xte_, yte_)
    }, temp)
    temp.flush()
    blob = storage.blob.Blob.from_string(storage_path, client=storage_client)
    blob.upload_from_filename(temp.name)
