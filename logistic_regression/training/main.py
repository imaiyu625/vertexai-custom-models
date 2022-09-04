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
args = parser.parse_args()

storage_client = storage.Client()

parsed_url = urlparse(args.input)
bucket = storage_client.bucket(parsed_url.netloc)
blob = bucket.blob(parsed_url.path.lstrip('/'))

def loss_func(w_, x_, y_):
    p_ = 1 / (1 + np.exp(-np.dot(x_, w_)))
    ll = np.dot(y_.T, np.log(p_)) + np.dot(1 - y_.T, np.log(1 - p_))
    return - ll / len(y_)

def fit(df):
    y_ = df[0].to_numpy()
    x_ = df.drop(0, axis=1).to_numpy()
    w_ = np.zeros(x_.shape[1])
    print('negative log likelihood:', loss_func(w_, x_, y_))
    wopt_ = optimize.minimize(loss_func, w_, args=(x_, y_), method='L-BFGS-B').x
    print('negative log likelihood:', loss_func(wopt_, x_, y_))
    return wopt_

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
