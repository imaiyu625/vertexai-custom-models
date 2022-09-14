# Delayed Feedback Model
see the [original paper](https://www.researchgate.net/publication/266660247_Modeling_delayed_feedback_in_display_advertising) for details

$\text{LogLikelihood} = \sum_{i,y_i=1} \log p(\mathbf{x}_i) + m(\mathbf{x}_i) \log \lambda(\mathbf{x}_i) + \log m(\mathbf{x}_i) + (m(\mathbf{x}_i)-1) \log d_i - (\lambda(\mathbf{x}_i)d_i)^{m(\mathbf{x}_i)} + \sum_{i,y_i=0} \log\left[1 - p(\mathbf{x}_i) + p(\mathbf{x}_i) \exp(-(\lambda(\mathbf{x}_i) e_i)^{m(\mathbf{x}_i})\right]$

where
$p(\mathbf{x}_i) = \frac{1}{1 + \exp(- \mathbf{w}_{\text{lr}}^{\top} \mathbf{x}_i)}$ means the probability of logistic regression,
$\lambda(\mathbf{x}_i) = \exp(\mathbf{w}_{\text{rate}}^{\top} \mathbf{x}_i)$ means the rate (inverse scale) parameter of Weibull distribution,
$m(\mathbf{x}_i) = \exp(\mathbf{w}_{\text{shape}}^{\top} \mathbf{x}_i)$ means the shape parameter of Weibull distribution

especially $m(\mathbf{x}_i) = 1$, then the same exponential distribution as in the original paper

# Preprocessing data
execute preprocess_criteo.py and upload data.csv to GCS

# Create model
## Push container image for training
same as [logistic_regression](https://github.com/imaiyu625/vertexai-custom-models/blob/main/logistic_regression/README.md#push-container-image-for-training)

## Execute custom training job on Vertex AI
```bash
DISPLAY_NAME=dfm-criteo
MACHINE_TYPE=n1-standard-4
REGION=YOUR_REGION
WORK_BUCKET=YOUR_GCS_BUCKET
WORK_PATH=YOUR_GCS_PATH

CONTAINER_IMAGE_URI=$(gcloud beta container images describe \
  gcr.io/${PROJECT_ID}/${CONTAINER_IMAGE_NAME}:latest \
  --format="value(image_summary.fully_qualified_digest)")

gcloud beta ai custom-jobs create \
  --display-name=$DISPLAY_NAME \
  --region=$REGION \
  --worker-pool-spec=machine-type=$MACHINE_TYPE,replica-count=1,container-image-uri=$CONTAINER_IMAGE_URI \
  --args="gs://${WORK_BUCKET}/${WORK_PATH}/data.csv","--output=gs://${WORK_BUCKET}/${WORK_PATH}/${DISPLAY_NAME}/"
```

# Deploy the created model
same as [logistic_regression](https://github.com/imaiyu625/vertexai-custom-models/blob/main/logistic_regression/README.md#deploy-the-created-model)

# Make endpoint
same as [logistic_regression](https://github.com/imaiyu625/vertexai-custom-models/blob/main/logistic_regression/README.md#make-endpoint)

## Online prediction
first column is the expected time from ts_click to ts_cv, and the second and subsequent columns are the features

```bash
curl -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  -d @online_prediction.jsonl \
  https://${REGION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${REGION}/endpoints/${ENDPOINT_ID}:predict

gcloud beta ai endpoints predict \
  $ENDPOINT_ID \
  --region=$REGION \
  --json-request=online_prediction.jsonl

gcloud beta ai endpoints raw-predict \
  $ENDPOINT_ID \
  --region=$REGION \
  --request='{"instances": [[0.1, 2, 3, -1, 0, 9, 2, 17, 8],[0.5, 2, 3, -1, 0, 9, 2, 17, 8]]}'
```

# Batch prediction
same as [logistic_regression](https://github.com/imaiyu625/vertexai-custom-models/blob/main/logistic_regression/README.md#batch-prediction)
