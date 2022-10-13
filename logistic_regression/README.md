# Preprocessing data
```bash
WORK_BUCKET=YOUR_GCS_BUCKET
WORK_PATH=YOUR_GCS_PATH

python3 preprocess_iris.py
gsutil cp data.csv gs://${WORK_BUCKET}/${WORK_PATH}/
```

# Create model
## Push container image for training
```bash
PROJECT_ID=$(gcloud config list project --format="value(core.project)")
REGION=YOUR_REGION
CONTAINER_IMAGE_NAME=YOUR_TRAINING_CONTAINER_IMAGE_NAME

gcloud builds submit \
  --region=${REGION} \
  --tag=gcr.io/${PROJECT_ID}/${CONTAINER_IMAGE_NAME} \
  ./training
```

## Execute custom training job on Vertex AI
```bash
DISPLAY_NAME=lr-iris
MACHINE_TYPE=n1-standard-4

CONTAINER_IMAGE_URI=$(gcloud beta container images describe \
  gcr.io/${PROJECT_ID}/${CONTAINER_IMAGE_NAME} \
  --format="value(image_summary.fully_qualified_digest)")

gcloud beta ai custom-jobs create \
  --display-name=$DISPLAY_NAME \
  --region=$REGION \
  --worker-pool-spec=machine-type=$MACHINE_TYPE,replica-count=1,container-image-uri=$CONTAINER_IMAGE_URI \
  --args="gs://${WORK_BUCKET}/${WORK_PATH}/data.csv","--output=gs://${WORK_BUCKET}/${WORK_PATH}/${DISPLAY_NAME}/"

# wait until status is JOB_STATE_SUCCEEDED
gcloud beta ai custom-jobs list \
  --filter="displayName:${DISPLAY_NAME}" \
  --region=$REGION \
  --sort-by=~createTime \
  --limit=1 \
  --format="value(state)"
```

# Deploy the created model
## Push container image for prediction
```bash
CONTAINER_IMAGE_NAME=YOUR_PREDICTION_CONTAINER_IMAGE_NAME

gcloud builds submit \
  --region=${REGION} \
  --tag=gcr.io/${PROJECT_ID}/${CONTAINER_IMAGE_NAME} \
  ./prediction
```

## Upload model artifact
```bash
CONTAINER_IMAGE_URI=$(gcloud beta container images describe \
  gcr.io/${PROJECT_ID}/${CONTAINER_IMAGE_NAME} \
  --format="value(image_summary.fully_qualified_digest)")

ARTIFACT_URI=$(gcloud beta ai custom-jobs list \
  --filter="displayName:${DISPLAY_NAME}" \
  --region=$REGION \
  --sort-by=~createTime \
  --limit=1 \
  --format='value(jobSpec.workerPoolSpecs.containerSpec.args[1])' \
  | sed -e 's/--output=//g')

gcloud beta ai models upload \
  --container-image-uri=$CONTAINER_IMAGE_URI \
  --display-name=$DISPLAY_NAME \
  --region=$REGION \
  --container-health-route=/health \
  --container-predict-route=/predict \
  --container-ports=8080 \
  --artifact-uri=$ARTIFACT_URI
```

# Make endpoint
```bash
MACHINE_TYPE=n1-standard-2

gcloud beta ai endpoints create \
  --display-name=$DISPLAY_NAME \
  --region=$REGION

MODEL_ID=$(gcloud beta ai models list \
  --filter="displayName:${DISPLAY_NAME}" \
  --region=$REGION \
  --sort-by=~createTime \
  --limit=1 \
  --format="value(MODEL_ID)")

ENDPOINT_ID=$(gcloud beta ai endpoints list \
  --filter="displayName:${DISPLAY_NAME}" \
  --region=$REGION \
  --sort-by=~createTime \
  --limit=1 \
  --format="value(ENDPOINT_ID)")

gcloud beta ai endpoints deploy-model \
  $ENDPOINT_ID \
  --display-name=$DISPLAY_NAME \
  --region=$REGION \
  --model=$MODEL_ID \
  --machine-type=$MACHINE_TYPE \
  --traffic-split=0=100 \
  --enable-access-logging \
  --enable-container-logging
```

## Online prediction
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
  --request='{"instances": [[5.1, 3.5, 1.4, 0.2],[5.7, 2.8, 4.1, 1.3]]}'
```

# Batch prediction
```bash
curl -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  -d @request.json \
  https://${REGION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${REGION}/batchPredictionJobs
```
