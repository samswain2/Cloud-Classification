# Import required modules
from pathlib import Path
import logging
import boto3
import botocore.exceptions

# Set up logging configuration
logger = logging.getLogger("clouds")


# Define function to upload artifacts to Amazon S3
def upload_artifacts(artifacts: Path, aws_config: dict) -> None:
    """Upload all artifacts in the specified directory to Amazon S3.

    Args:
        artifacts (Path): The directory containing all the artifacts from a given experiment.
        aws_config (dict): Configuration required to upload artifacts to S3; see example config file for structure.

    Returns:
        None
    """

    # Log start of upload process
    logger.info("Starting to upload artifacts to S3...")

    # Extract required information from aws_config dictionary
    region = aws_config.get("region")
    bucket = aws_config.get("bucket_name")
    prefix = aws_config.get("prefix")

    try:
        # Create S3 client object
        s3_client = boto3.client("s3", region_name=region)

    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        logger.error("Error occurred while creating S3 client: %s", e)
        return

    # Log details of bucket and prefix for uploaded files
    logger.info("Uploading artifacts to bucket %s with prefix %s", bucket, prefix)

    # Upload each file in the artifacts directory to S3
    for file in artifacts.glob("**/*"):
        if file.is_file():
            try:
                # Upload file to S3 using the client object
                s3_client.upload_file(
                    str(file),
                    bucket,
                    f"{prefix}/{file.relative_to(artifacts)}",
                )
                # Log successful upload
                logger.info("Uploaded %s to S3", file)
            except (
                botocore.exceptions.BotoCoreError,
                botocore.exceptions.ClientError,
            ) as e:
                # Log error if file upload fails
                logger.error("Error occurred while uploading %s to S3: %s", file, e)

    # Log completion of upload process
    logger.info("Finished uploading artifacts to S3")
