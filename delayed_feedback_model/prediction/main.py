import os
import pickle
from google.cloud import storage
from urllib.parse import urlparse
import numpy as np
from fastapi import Request, FastAPI

artifact_filename = 'model.pkl'
model_directory = os.environ['AIP_STORAGE_URI']
storage_path = os.path.join(model_directory, artifact_filename)

storage_client = storage.Client()

parsed_url = urlparse(storage_path)
bucket = storage_client.bucket(parsed_url.netloc)
blob = bucket.blob(parsed_url.path.lstrip('/'))
param = pickle.loads(blob.download_as_string())

app = FastAPI()

AIP_HEALTH_ROUTE = os.environ.get('AIP_HEALTH_ROUTE', '/health')
AIP_PREDICT_ROUTE = os.environ.get('AIP_PREDICT_ROUTE', '/predict')

@app.get(AIP_HEALTH_ROUTE)
async def health():
    return 200

def delayed_feedback_model(x_):
    t_, x_ = np.hsplit(x_, [1])
    x_ = np.concatenate((np.ones((x_.shape[0], 1)), x_), axis=1) # add bias term
    wx_ = np.dot(param, x_.T)
    p_ = 1 / (1 + np.exp(-wx_[0]))
    l_ = np.exp(wx_[1])
    m_ = np.exp(wx_[2])
    return p_ * (1 - np.exp(-(l_ * t_.T)**m_))

@app.post(AIP_PREDICT_ROUTE)
async def predict(request: Request):
    body = await request.json()
    response = delayed_feedback_model(np.array(body['instances']))
    if isinstance(response, np.ndarray):
        return {'predictions': response.tolist()}
    else:
        return 500
