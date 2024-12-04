import boto3
import sagemaker
import sagemaker.session
from sagemaker.workflow.pipeline_context import PipelineSession
from dotenv import load_dotenv
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
import os
import logging
# Processing step
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep
# Training step
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TrainingStep
# Evaluation step
from sagemaker.processing import ScriptProcessor
from sagemaker.workflow.properties import PropertyFile
# Register model
from sagemaker.model_metrics import MetricsSource, ModelMetrics 
from sagemaker.workflow.step_collections import RegisterModel
# Pipeline definition
from sagemaker.workflow.pipeline import Pipeline
import json
# Conditional step
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet


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

# Define pipeline parameters

processing_instance_count = ParameterInteger(
    name="ProcessingInstanceCount",
    default_value=1
)
input_data = ParameterString(
    name="InputData",
    default_value=input_data_uri,
)

model_approval_status = ParameterString(
    name="ModelApprovalStatus",
    default_value="PendingManualApproval"
)

# Processing step

framework_version = "0.23-1"

sklearn_processor = SKLearnProcessor(
    framework_version=framework_version,
    instance_type="ml.t3.medium",
    instance_count=processing_instance_count,
    base_job_name="sklearn-abalone-process",
    sagemaker_session=pipeline_session,
    role=role,
)  

processor_args = sklearn_processor.run(
    inputs=[
      ProcessingInput(source=input_data, destination="/opt/ml/processing/input"),  
    ],
    outputs=[
        ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
        ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
        ProcessingOutput(output_name="test", source="/opt/ml/processing/test")
    ],
    code="abalone/preprocessing.py",
) 

step_process = ProcessingStep(
    name="AbaloneProcess",
    step_args=processor_args
)

# Training step

model_path = f"s3://{default_bucket}/AbaloneTrain"

image_uri = sagemaker.image_uris.retrieve(
    framework="xgboost",
    region=region,
    version="1.0-1",
    py_version="py3",
    instance_type="ml.m5.xlarge"
)
xgb_train = Estimator(
    image_uri=image_uri,
    instance_type="ml.m5.xlarge",
    instance_count=1,
    output_path=model_path,
    sagemaker_session=pipeline_session,
    role=role,
)
xgb_train.set_hyperparameters(
    objective="reg:linear",
    num_round=50,
    max_depth=4,
    eta=0.2,
    gamma=3,
    min_child_weight=6,
    subsample=0.7,
    silent=0
)

train_args = xgb_train.fit(
    inputs={
        "train": TrainingInput(
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                "train"
            ].S3Output.S3Uri,
            content_type="text/csv"
        ),
        "validation": TrainingInput(
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                "validation"
            ].S3Output.S3Uri,
            content_type="text/csv"
        )
    },
)

step_train = TrainingStep(
    name="AbaloneTrain",
    step_args = train_args
)

# Evaluation step

script_eval = ScriptProcessor(
    image_uri=image_uri,
    command=["python3"],
    instance_type="ml.t3.medium",
    instance_count=1,
    base_job_name="script-abalone-eval",
    sagemaker_session=pipeline_session,
    role=role,
)

evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation",
    path="evaluation.json"
)

eval_args = script_eval.run(
        inputs=[
        ProcessingInput(
            source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model"
        ),
        ProcessingInput(
            source=step_process.properties.ProcessingOutputConfig.Outputs[
                "test"
            ].S3Output.S3Uri,
            destination="/opt/ml/processing/test"
        )
    ],
    outputs=[
        ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
    ],
    code="abalone/evaluation.py",
)

step_eval = ProcessingStep(
    name="AbaloneEval",
    step_args=eval_args,
    property_files=[evaluation_report],
)

# Register model

model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri="{}/evaluation.json".format(
            step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
        ),
        content_type="application/json"
    )
)
step_register = RegisterModel(
    name="AbaloneRegisterModel",
    estimator=xgb_train,
    model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
    transform_instances=["ml.m5.large"],
    model_package_group_name=model_package_group_name,
    approval_status=model_approval_status,
    model_metrics=model_metrics
)

# Conditional step

cond_lte = ConditionLessThanOrEqualTo(
    left=JsonGet(
        step_name=step_eval.name,
        property_file=evaluation_report,
        json_path="regression_metrics.mse.value"
    ),
    right=6.0,
)
step_cond = ConditionStep(
    name="CheckMSEAbaloneEvaluation",
    conditions=[cond_lte],
    if_steps=[step_register],
    else_steps=[],
)

# Create pipeline

pipeline_name = f"AbalonePipeline"
pipeline = Pipeline(
    name=pipeline_name,
    parameters=[
        processing_instance_count,
        input_data,
        model_approval_status,
    ],
    steps=[step_process, step_train, step_eval, step_cond],
)

pipeline.upsert(role_arn=role)

json.loads(pipeline.definition())

execution = pipeline.start()
execution.describe()