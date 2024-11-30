import boto3
import sagemaker
import sagemaker.session
from sagemaker.workflow.pipeline_context import PipelineSession
from dotenv import load_dotenv
import os

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
