# Cloud Classification Project

## Overview
The main intent of the cloud classification pipeline is to create a reproducible and modular machine learning process that can classify clouds into one of two types based on features generated from cloud images. The pipeline is designed to be easily run in different environments by leveraging containerization and following programming best practices. (Link to cloud data: https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/taylor/cloud.data)

The pipeline consists of several stages, including:

1. Configuration Management (YAML configuration)

2. Data Preprocessing

3. Model Training

4. Model Evaluation

5. Artifact Generation

6. Artifact Uploading (AWS S3)

7. Unit Testing

The pipeline is designed to be run with a single command inside a Docker container, ensuring a consistent and portable environment for executing the pipeline across different systems. 

## Build Docker pipeline image

1. Download this repository.

2. Install Docker: If Docker is not already installed on your machine, you need to install it first. Follow the instructions provided by Docker to install it on your operating system. (Docker Desktop: https://www.docker.com/products/docker-desktop/)

3. Open the Docker Desktop application.

4. Open a terminal (or command prompt on Windows) and navigate to the main project directory `423-assignment-2-sms5736/`.

5. Configure the `default-config.yaml` file located in the `./config/` directory according to your specifications. **This is important if you want to upload artifacts to your AWS S3 bucket.**

6. Build the Docker pipeline image by running the following command:
    ```bash
    docker build -t hw02_pipeline_image -f dockerfiles/docker_pipeline/Dockerfile .
    ```
7. If you have enabled uploads to AWS S3, you will need your AWS account credentials.
    1. Navigate to your AWS start page.
    2. Select your AWS account.
    3. Select `Command line or programmatic access`.
    4. Copy and paste the following commands into your terminal or command prompt, replacing <keys> with your AWS keys:
    - For Mac/Linux:
        - `export AWS_ACCESS_KEY_ID="<access_key>"`
        - `export AWS_SECRET_ACCESS_KEY="<secret_access_key>"`
        - `export AWS_SESSION_TOKEN="<session_token>"`
    - For Windows:
        - `set AWS_ACCESS_KEY_ID="<access_key>"`
        - `set AWS_SECRET_ACCESS_KEY="<secret_access_key>"`
        - `set AWS_SESSION_TOKEN="<session_token>"`
    5. Paste the copied keys into your terminal and run

8. Replace DIR in the command below with the path where you want the run artifacts to be placed:
```bash
docker run -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY -e AWS_SESSION_TOKEN=$AWS_SESSION_TOKEN -e RUN_NAME=pipeline_container -v "DIR:/hw02/runs" hw02_pipeline_image
```

9. Run the command. The file will save the artifacts to your desired location.

## Build Docker unit test image

1. Download this repository if you haven't done so already.

2. Install Docker: If Docker is not already installed on your machine, you need to install it first. Follow the instructions provided by Docker to install it on your operating system. (Docker Desktop: https://www.docker.com/products/docker-desktop/)

3. Open the Docker Desktop application.

4. Open a terminal (or command prompt on Windows) and navigate to the main project directory `423-assignment-2-sms5736/`.

5. Build the Docker unit test image by running the following command:
```bash
docker build -t hw02_unittest_image -f dockerfiles/docker_unittest/Dockerfile .
```

6. Run the unit tests by executing the following command and the results will appear in your terminal:

```bash
docker run hw02_unittest_image
```
