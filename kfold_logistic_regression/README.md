# Preprocessing data
```bash
WORK_BUCKET=YOUR_GCS_BUCKET
WORK_PATH=YOUR_GCS_PATH

python3 ../logistic_regression/preprocess_iris.py
gsutil cp data.csv gs://${WORK_BUCKET}/${WORK_PATH}/
```

# Create model
## Push container image for training
same as [logistic_regression](https://github.com/imaiyu625/vertexai-custom-models/blob/main/logistic_regression/README.md#push-container-image-for-training)

## Make training config
```bash
DISPLAY_NAME=kfold-lr-iris
N_SPLITS=YOUR_NUMBER_SPLITS # at least 2
N_PARALLELISM=YOUR_NUMBER_PARALLELISM # multiple VMs
RANDOM_STATE=YOUR_RANDOM_STATE # controls the randomness of each fold

printf $(cat job.json | jq -c) \
  gcr.io/${PROJECT_ID}/${CONTAINER_IMAGE_NAME} \
  gs://${WORK_BUCKET}/${WORK_PATH}/data.csv \
  gs://${WORK_BUCKET}/${WORK_PATH}/${DISPLAY_NAME}/ \
  $RANDOM_STATE \
  $N_SPLITS \
  $N_PARALLELISM > job_o.json
```

## Execute k-fold custom training job on Cloud Batch
```bash
LOCATION=YOUR_LOCATION # https://cloud.google.com/batch/docs/locations#regions

gcloud beta batch jobs submit $DISPLAY_NAME \
  --location=$LOCATION \
  --config=job_o.json
```

# Deploy the created model
## Push container image for prediction
```bash
CONTAINER_IMAGE_NAME=YOUR_PREDICTION_CONTAINER_IMAGE_NAME

gcloud builds submit \
  --region=${REGION} \
  --tag=gcr.io/${PROJECT_ID}/${CONTAINER_IMAGE_NAME} \
  ../logistic_regression/prediction
```

## Upload model artifact
```bash
JOB_INDEX=0 # 0 ~ N_SPLITS-1

CONTAINER_IMAGE_URI=$(gcloud beta container images describe \
  gcr.io/${PROJECT_ID}/${CONTAINER_IMAGE_NAME} \
  --format="value(image_summary.fully_qualified_digest)")

gcloud beta ai models upload \
  --container-image-uri=$CONTAINER_IMAGE_URI \
  --display-name=$DISPLAY_NAME \
  --region=$REGION \
  --container-health-route=/health \
  --container-predict-route=/predict \
  --container-ports=8080 \
  --artifact-uri=gs://${WORK_BUCKET}/${WORK_PATH}/${DISPLAY_NAME}/${JOB_INDEX}/
```

# Make endpoint
same as [logistic_regression](https://github.com/imaiyu625/vertexai-custom-models/blob/main/logistic_regression/README.md#make-endpoint)

# Batch prediction
same as [logistic_regression](https://github.com/imaiyu625/vertexai-custom-models/blob/main/logistic_regression/README.md#batch-prediction)
