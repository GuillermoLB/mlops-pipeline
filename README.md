# mlops-pipeline

This project demonstrates an MLOps pipeline using AWS SageMaker and Python SDK. The pipeline includes data preprocessing, model training, evaluation, and deployment steps.

## Setup

1. **Clone the repository:**

    ```sh
    git clone <repository-url>
    cd mlops-pipeline
    ```

2. **Create and activate a virtual environment:**

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages:**

    ```sh
    pip install -r requirements.txt
    ```

4. **Set up AWS credentials:**

    Update the `.env` file with your AWS credentials and region:

    ```env
    AWS_ACCESS_KEY_ID=your_access_key_id
    AWS_SECRET_ACCESS_KEY=your_secret_access_key
    ROLE=your_sagemaker_execution_role
    AWS_REGION=your_aws_region
    ```

## Usage

1. **Run the pipeline:**

    You can run the pipeline by executing the `pipeline.ipynb` notebook or running the `pipeline.py` script.

    ```sh
    python abalone/pipeline.py
    ```

2. **Preprocessing:**

    The preprocessing script `preprocessing.py` handles data cleaning, normalization, and splitting into training, validation, and test sets.

3. **Training:**

    The training step uses the preprocessed data to train a model using SageMaker's XGBoost algorithm.

4. **Evaluation:**

    The evaluation script `evaluation.py` evaluates the trained model and generates a report.

## License

This project is licensed under the MIT License.