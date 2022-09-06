import os
import pickle
import tempfile
from google.cloud import storage
from urllib.parse import urlparse
from io import BytesIO
import argparse
import pandas as pd
import numpy as np
import scipy.optimize as optimize

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str)
parser.add_argument('--output', type=str)
parser.add_argument('--nfactors', type=int, default=8)
args = parser.parse_args()

storage_client = storage.Client()

parsed_url = urlparse(args.input)
bucket = storage_client.bucket(parsed_url.netloc)
blob = bucket.blob(parsed_url.path.lstrip('/'))

def loss_func(w_, x_, y_):
    w_, v_ = np.hsplit(w_.reshape((x_.shape[1], args.nfactors + 1)), [1])
    v_ = np.vsplit(v_, [1])[1] # remove bias term
    wx_ = np.dot(x_, w_.T.flatten())

    v_ij = np.triu(np.dot(v_, v_.T), k=1)

    vx_ = list()
    for x in x_.reshape(x_.shape[0], 1, x_.shape[1]):
        x = np.hsplit(x, [1])[1] # remove bias term
        x_ij = np.tril(np.dot(x.T, x), k=-1)
        vx_.append(np.diag(np.dot(v_ij, x_ij)).sum())
    p_ = 1 / (1 + np.exp(-(wx_ + vx_)))
    ll = np.dot(y_.T, np.log(p_)) + np.dot(1 - y_.T, np.log(1 - p_))
    return - ll / len(y_)

def fit(df):
    y_ = df[0].to_numpy()
    x_ = df.drop(0, axis=1).to_numpy()
    x_ = np.concatenate((np.ones((x_.shape[0], 1)), x_), axis=1) # add bias term
    w_ = np.zeros(x_.shape[1] * (args.nfactors + 1)) + 1e-3
    print('negative log likelihood:', loss_func(w_, x_, y_))
    wopt_ = optimize.minimize(loss_func, w_, args=(x_, y_), method='L-BFGS-B').x
    print('negative log likelihood:', loss_func(wopt_, x_, y_))
    return wopt_.reshape((x_.shape[1], args.nfactors + 1))

raw_data = pd.read_csv(BytesIO(blob.download_as_bytes()), header=None)
param = fit(raw_data)

model_directory = os.environ.get('AIP_MODEL_DIR', args.output)
artifact_filename = 'model.pkl'
storage_path = os.path.join(model_directory, artifact_filename)

with tempfile.NamedTemporaryFile(mode='wb') as temp:
    pickle.dump(param, temp)
    temp.flush()
    blob = storage.blob.Blob.from_string(storage_path, client=storage_client)
    blob.upload_from_filename(temp.name)
