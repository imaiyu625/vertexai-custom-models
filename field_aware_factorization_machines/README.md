# Field-aware Factorization Machines
see the [original paper](https://www.researchgate.net/publication/307573604_Field-aware_Factorization_Machines_for_CTR_Prediction) for details

# Preprocessing data
same as [logistic_regression](https://github.com/imaiyu625/vertexai-custom-models/blob/main/logistic_regression/README.md#preprocessing-data)

# Create model
## Push container image for training
same as [logistic_regression](https://github.com/imaiyu625/vertexai-custom-models/blob/main/logistic_regression/README.md#push-container-image-for-training)

## Execute custom training job on Vertex AI
```bash
DISPLAY_NAME=ffm-iris
MACHINE_TYPE=n1-standard-4
REGION=YOUR_REGION
WORK_BUCKET=YOUR_GCS_BUCKET
WORK_PATH=YOUR_GCS_PATH
NUM_FACTORS=DIMENSIONALITY_OF_FACTORIZATION

CONTAINER_IMAGE_URI=$(gcloud beta container images describe \
  gcr.io/${PROJECT_ID}/${CONTAINER_IMAGE_NAME}:latest \
  --format="value(image_summary.fully_qualified_digest)")

gcloud beta ai custom-jobs create \
  --display-name=$DISPLAY_NAME \
  --region=$REGION \
  --worker-pool-spec=machine-type=$MACHINE_TYPE,replica-count=1,container-image-uri=$CONTAINER_IMAGE_URI \
  --args="gs://${WORK_BUCKET}/${WORK_PATH}/data.csv","--output=gs://${WORK_BUCKET}/${WORK_PATH}/${DISPLAY_NAME}/","--nfactors=${NUM_FACTORS}"
```

# Deploy the created model
same as [logistic_regression](https://github.com/imaiyu625/vertexai-custom-models/blob/main/logistic_regression/README.md#deploy-the-created-model)

# Make endpoint
same as [logistic_regression](https://github.com/imaiyu625/vertexai-custom-models/blob/main/logistic_regression/README.md#make-endpoint)

# Batch prediction
same as [logistic_regression](https://github.com/imaiyu625/vertexai-custom-models/blob/main/logistic_regression/README.md#batch-prediction)
