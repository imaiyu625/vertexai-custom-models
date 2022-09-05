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

def loss_func(w_, x_, y_, t_):
    wx_ = np.dot(w_.reshape((3, x_.shape[1])), x_.T)
    p_ = 1 / (1 + np.exp(-wx_[0]))
    l_ = np.exp(wx_[1])
    m_ = np.exp(wx_[2])
    ll = (y_ * (np.log(p_) + np.log(l_) + np.log(m_) + (m_ - 1) * np.log(l_ * t_) - (l_ * t_)**m_)).sum()
    ll += ((1 - y_) * np.log(1 - p_ + p_ * np.exp(-(l_ * t_)**m_))).sum()
    return - ll / len(y_)

def fit(df):
    y_ = df[0].to_numpy()
    t_ = df[1].to_numpy() + 1e-10
    x_ = df.drop([0,1], axis=1).to_numpy()
    x_ = np.concatenate((np.ones((x_.shape[0], 1)), x_), axis=1) # add bias term
    w_ = np.zeros(3 * x_.shape[1])
    print('negative log likelihood:', loss_func(w_, x_, y_, t_))
    wopt_ = optimize.minimize(loss_func, w_, args=(x_, y_, t_), method='L-BFGS-B').x
    print('negative log likelihood:', loss_func(wopt_, x_, y_, t_))
    return wopt_.reshape((3, x_.shape[1]))

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
