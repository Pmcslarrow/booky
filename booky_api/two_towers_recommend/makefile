#
# Import .env
#
ifneq (,$(wildcard ./.env))
    include .env
    export
endif

#
# Globals
#
LOCATION = $(ENV_LOCATION)
PROJECT_ID = $(ENV_PROJECT_ID)
REPOSITORY = $(ENV_REPOSITORY)
IMAGE = $(ENV_IMAGE)
RUN_ID = $(ENV_RUN_ID)
ARTIFACT_URI := $(LOCATION)-docker.pkg.dev/$(PROJECT_ID)/$(REPOSITORY)/$(IMAGE):latest

build:
	docker build --platform linux/amd64 -t booky-image:latest .

run: 
	docker run --rm -p 8080:8080 booky-image:latest


tag:
	docker tag $(IMAGE) $(ARTIFACT_URI)

push:
	docker push $(ARTIFACT_URI)

deploy:
	gcloud ai models upload \
		--region=$(LOCATION) \
		--display-name=booky-model \
		--artifact-uri=gs://booky-bucket/runs/$(RUN_ID) \
		--container-image-uri=$(ARTIFACT_URI)

full:
	$(MAKE) build
	$(MAKE) tag 
	$(MAKE) push 
	$(MAKE) deploy

#
# NOTE: Future CLI integrations
#
# # Create endpoint
# 	gcloud ai endpoints create \
# 	--region=$(LOCATION) \
# 	--display-name=booky-endpoint
#
# # Deploy to endpoint
# gcloud ai endpoints deploy-model $(ENDPOINT_ID) \
# 	--region=$(LOCATION) \
# 	--model=$(MODEL_ID) \
# 	--display-name=booky-deployment \
# 	--machine-type=n1-standard-4 \
# 	--min-replica-count=1 \
# 	--max-replica-count=1 \
# 	--traffic-split=0=100
#
# IAM Policy for Model registry to read from here
# gcloud artifacts repositories add-iam-policy-binding REPOSITORY_NAME \
#     --location=REGION \
#     --member="serviceAccount:YOUR_SERVICE_ACCOUNT_EMAIL" \
#     --role="roles/artifactregistry.reader"