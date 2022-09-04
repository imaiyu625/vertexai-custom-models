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

def log_reg(x_):
    return 1 / (1 + np.exp(-np.dot(x_, param)))

@app.post(AIP_PREDICT_ROUTE)
async def predict(request: Request):
    body = await request.json()
    response = log_reg(np.array(body['instances']))
    if isinstance(response, np.ndarray):
        return {'predictions': response.tolist()}
    else:
        return 500
