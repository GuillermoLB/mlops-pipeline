import boto3
import sagemaker
import sagemaker.session
from sagemaker.workflow.pipeline_context import PipelineSession
from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
region = os.getenv("AWS_REGION")
role = os.getenv("ROLE")

# Set environment variables explicitly
os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key_id
os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_access_key
os.environ['AWS_DEFAULT_REGION'] = region

# Create a boto3 session with the loaded credentials
boto_session = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region
)

sagemaker_session = sagemaker.session.Session(boto_session)
default_bucket = sagemaker_session.default_bucket()

pipeline_session = PipelineSession()

model_package_group_name = f"AbaloneModelPackageGroupName"

# Download the dataset
local_path = "data/abalone-dataset.csv"

# Ensure the directory exists
os.makedirs(os.path.dirname(local_path), exist_ok=True)
logger.info(f"Directory created: {os.path.dirname(local_path)}")

s3 = boto3.resource("s3")
bucket_name = f"sagemaker-servicecatalog-seedcode-{region}"
logger.info(f"Downloading file from bucket: {bucket_name}")

try:
    s3.Bucket(bucket_name).download_file(
        "dataset/abalone-dataset.csv",
        local_path
    )
    logger.info(f"File downloaded successfully to {local_path}")
except Exception as e:
    logger.error(f"Error downloading file: {e}")

# Verify if the file exists
if os.path.exists(local_path):
    logger.info(f"File exists at {local_path}")
else:
    logger.error(f"File does not exist at {local_path}")

base_uri = f"s3://{default_bucket}/abalone"
input_data_uri = sagemaker.s3.S3Uploader.upload(
    local_path=local_path, 
    desired_s3_uri=base_uri,
)
print(input_data_uri)
