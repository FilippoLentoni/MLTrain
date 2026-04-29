# MLTrain — SageMaker XGBoost Pipeline (AWS CDK, TypeScript)

A portable, account-agnostic AWS CDK (TypeScript) stack that provisions an end-to-end SageMaker pipeline:

1. **Preprocess** (Processing job, scikit-learn image) — generates synthetic binary-classification data with `sklearn.datasets.make_classification`, splits into `train` / `validation` / `test`, writes CSVs to S3.
2. **Train** (Training job, XGBoost framework container, **script mode**) — runs your own [scripts/train.py](scripts/train.py) inside the AWS-provided XGBoost framework container; it imports `xgboost` + `numpy`, calls `xgb.train(...)`, saves `model.tar.gz` to S3. **No custom Docker image required** — your script is bundled into a `sourcedir.tar.gz` at synth time and downloaded into the container by the framework.
3. **Postprocess / Evaluate** (Processing job, XGBoost image) — loads the trained model and the held-out test set, computes accuracy, writes `metrics.json` to S3.

Built on native CDK constructs (`aws_s3`, `aws_iam`, `aws_s3_deployment`, `aws_sagemaker.CfnPipeline`). The SageMaker built-in image account is resolved per-region via a `CfnMapping`, so the same code deploys cleanly into any of 16 AWS regions without edits.

> **Two languages on purpose.** The CDK app is TypeScript. The ML scripts ([scripts/preprocess.py](scripts/preprocess.py), [scripts/postprocess.py](scripts/postprocess.py)) are Python, because they execute *inside* SageMaker's Python-based scikit-learn and XGBoost containers. CDK only ships them to S3 and tells SageMaker to run them — the language of your CDK app is independent of the language of your ML code.

---

## Repository layout

```
MLTrain/
├── lib/
│   └── ml-train-stack.ts        # Stack + app entry: bucket, role, scripts upload, CfnPipeline
├── scripts/                     # Uploaded to S3 and executed inside SageMaker containers
│   ├── preprocess.py            # Synthetic data + train/val/test split → CSV
│   ├── train.py                 # Custom XGBoost training (script mode)
│   └── postprocess.py           # Load model + test set → accuracy → metrics.json
├── cdk.json                     # CDK CLI config
├── package.json                 # npm deps (aws-cdk-lib, ts-node, typescript)
└── tsconfig.json                # TypeScript config
```

---

## Prerequisites

| Tool        | Version       | Notes                                                    |
|-------------|---------------|----------------------------------------------------------|
| Node.js     | 20+ (LTS)     | required by CDK CLI and the TypeScript app               |
| npm         | 10+           | bundled with Node                                        |
| AWS CDK CLI | 2.130+        | `sudo npm install -g aws-cdk`                            |
| AWS CLI     | 2.x           | for credentials and triggering the pipeline              |

> Node 18 still works but is end-of-life — install Node 20 LTS to silence the CDK warning.
> No Python venv needed for *deploying*. Python is used only inside SageMaker's containers.

---

## 1. AWS account setup

The stack works in any AWS account/region pair (within the 16 regions covered by the `SAGEMAKER_IMAGE_ACCOUNTS` map in [lib/ml-train-stack.ts](lib/ml-train-stack.ts)).

### 1a. Identity used to **deploy**

Whoever runs `cdk deploy` needs CloudFormation + the right service permissions. The simplest setup is an IAM user/role with `AdministratorAccess` (deployment-only). For a tighter policy you need at minimum:

- `cloudformation:*` on the stack
- `s3:*` on the bootstrap + pipeline buckets
- `iam:CreateRole`, `iam:AttachRolePolicy`, `iam:PutRolePolicy`, `iam:PassRole`, `iam:CreateServiceLinkedRole`
- `sagemaker:CreatePipeline`, `sagemaker:UpdatePipeline`, `sagemaker:DeletePipeline`, `sagemaker:DescribePipeline`
- `lambda:*` and `logs:*` (for the `BucketDeployment` custom resource)
- `ssm:GetParameter` (for the CDK bootstrap version SSM parameter)

### 1b. Identity used to **run** the pipeline

To trigger an execution after deploy you need:

- `sagemaker:StartPipelineExecution`
- `sagemaker:DescribePipelineExecution`, `sagemaker:ListPipelineExecutionSteps` (to monitor)

The pipeline itself runs under a SageMaker execution role created by the stack (`SageMakerExecutionRole`); it's granted `AmazonSageMakerFullAccess` plus read/write on the pipeline bucket. You don't manage it manually.

### 1c. Configure credentials

Pick one:

```bash
# Option A — environment variables
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...      # if using STS
export AWS_DEFAULT_REGION=us-east-1

# Option B — named profile
aws configure --profile mlpipeline
export AWS_PROFILE=mlpipeline
export AWS_DEFAULT_REGION=us-east-1

# Option C — EC2/ECS/Lambda instance role (no config needed)
```

Verify:

```bash
aws sts get-caller-identity
```

---

## 2. Local environment setup

```bash
git clone <this repo> MLTrain && cd MLTrain

# Node + CDK CLI (one-time, machine-wide)
sudo dnf install -y nodejs npm        # Amazon Linux 2023
# or: brew install node               # macOS
# or: nvm install --lts
sudo npm install -g aws-cdk
cdk --version    # expect 2.x

# Project-local TypeScript deps
npm install
```

---

## 3. Bootstrap the AWS environment (once per account/region)

CDK needs a small set of helper resources (S3 staging bucket, IAM roles) in every account/region you deploy to:

```bash
cdk bootstrap aws://<ACCOUNT_ID>/<REGION>
# example
cdk bootstrap aws://123456789012/us-east-1
```

You only do this once per account+region.

---

## 4. Deploy the stack

```bash
cdk synth                              # optional — preview the CloudFormation template
cdk deploy --require-approval never
```

Stack outputs (printed after deploy):

| Output             | Meaning                                                  |
|--------------------|----------------------------------------------------------|
| `BucketName`       | S3 bucket holding scripts, datasets, model, metrics      |
| `PipelineName`     | SageMaker Pipeline name (`xgboost-pipeline-MLTrainStack`)|
| `ExecutionRoleArn` | SageMaker execution role used by the pipeline            |
| `StartCommand`     | Ready-to-run AWS CLI command to trigger an execution     |

---

## 5. Run the pipeline

```bash
aws sagemaker start-pipeline-execution \
    --pipeline-name xgboost-pipeline-MLTrainStack
```

Or, in the AWS Console: **SageMaker Studio → Pipelines → xgboost-pipeline-MLTrainStack → Create execution**.

End-to-end runtime is roughly **8–12 minutes** on `ml.m5.xlarge` instances (≈ 3 min preprocess + 3–5 min train + 2 min evaluate).

### Watch progress

```bash
EXEC_ARN=$(aws sagemaker start-pipeline-execution \
    --pipeline-name xgboost-pipeline-MLTrainStack \
    --query PipelineExecutionArn --output text)

aws sagemaker list-pipeline-execution-steps --pipeline-execution-arn "$EXEC_ARN"
```

### Inspect the metrics

```bash
BUCKET=$(aws cloudformation describe-stacks --stack-name MLTrainStack \
    --query "Stacks[0].Outputs[?OutputKey=='BucketName'].OutputValue" --output text)

aws s3 cp s3://$BUCKET/pipeline/postprocess/metrics/metrics.json - | jq
```

Expected shape:

```json
{
  "binary_classification_metrics": {
    "accuracy": { "value": 0.93..., "standard_deviation": "NaN" }
  },
  "n_samples": 1500
}
```

---

## 6. S3 layout produced by the pipeline

```
s3://<BucketName>/
├── code/                                        # Processing-job scripts (ScriptsDeployment)
│   ├── preprocess.py
│   ├── train.py
│   └── postprocess.py
├── sourcedir/
│   └── sourcedir.tar.gz                         # train.py packaged for script-mode training
└── pipeline/
    ├── preprocess/
    │   ├── train/train.csv
    │   ├── validation/validation.csv
    │   └── test/test.csv
    ├── training/<job-name>/output/model.tar.gz
    └── postprocess/metrics/metrics.json
```

---

## 7. Where to change what

| You want to change…              | Edit this file                                              |
|----------------------------------|-------------------------------------------------------------|
| AWS infrastructure (bucket, IAM, pipeline DAG, instance types) | [lib/ml-train-stack.ts](lib/ml-train-stack.ts) |
| XGBoost hyperparameters / pipeline-level training config | `HyperParameters` block in [lib/ml-train-stack.ts](lib/ml-train-stack.ts) |
| **Training algorithm itself** (`xgb.train` call, custom losses, callbacks) | [scripts/train.py](scripts/train.py) |
| Synthetic data generation / preprocessing logic | [scripts/preprocess.py](scripts/preprocess.py) |
| Evaluation metric / model loading logic        | [scripts/postprocess.py](scripts/postprocess.py) |
| Supported regions                | `SAGEMAKER_IMAGE_ACCOUNTS` map in [lib/ml-train-stack.ts](lib/ml-train-stack.ts) — extend with the [official region/account list](https://docs.aws.amazon.com/sagemaker/latest/dg-ecr-paths/sagemaker-algo-docker-registry-paths.html) |

Pipeline parameters `ProcessingInstanceType` / `TrainingInstanceType` default to `ml.m5.xlarge`. Override per-execution:

```bash
aws sagemaker start-pipeline-execution \
    --pipeline-name xgboost-pipeline-MLTrainStack \
    --pipeline-parameters Name=TrainingInstanceType,Value=ml.m5.2xlarge
```

---

## 7b. Bring-your-own-model spectrum (script mode vs BYOC)

| Mode                            | Who writes the training code     | Need to build a Docker image? | Used here |
|---------------------------------|----------------------------------|-------------------------------|-----------|
| Built-in algorithm              | Nobody — config only             | No                            | No        |
| **Script mode** *(this stack)*  | You — `scripts/train.py`         | **No** — AWS provides the framework container | **Yes** |
| Bring Your Own Container (BYOC) | You — `train.py` + `Dockerfile`  | Yes (`docker build` + ECR push) | No      |

In script mode, the same image (`sagemaker-xgboost:1.7-1`) is used as built-in mode, but the framework's container entrypoint detects script mode via the special `sagemaker_program` hyperparameter and runs:

```bash
python /opt/ml/code/train.py --num_round 100 --max_depth 5 --eta 0.2 ...
```

The `sagemaker_submit_directory` hyperparameter points to a `.tar.gz` in S3 containing your script(s); the framework downloads it into the container at runtime.

**To switch to BYOC** (e.g. you need a custom Python dependency the AWS image doesn't have):
1. Write a `Dockerfile` that extends `sagemaker-xgboost:1.7-1` and `pip install`s extras.
2. Build and push to ECR.
3. In [lib/ml-train-stack.ts](lib/ml-train-stack.ts), replace `xgboostImage` with the ECR URI of your image.
4. Drop the `sagemaker_*` hyperparameters and use a custom `ContainerEntrypoint` if your image's entrypoint differs.

---

## 8. Tear down

```bash
cdk destroy
```

Removes the stack, the bucket (objects auto-deleted), the role, and the pipeline. The CDK bootstrap stack (`CDKToolkit`) stays in place — it's reusable for any other CDK app in the same account+region.

---

## 9. Portability checklist

- ✅ No hardcoded AWS account ID — `CDK_DEFAULT_ACCOUNT` (or env detection) drives it.
- ✅ No hardcoded region — `CDK_DEFAULT_REGION` (or env detection) drives it.
- ✅ No hardcoded bucket name — auto-generated by CloudFormation.
- ✅ SageMaker image registry account resolved via `CfnMapping` keyed on `AWS::Region`.
- ✅ All IAM resources are stack-local; nothing references external roles.
- ✅ Scripts are uploaded as a CDK asset, not pulled from an external location.

To move to another account: configure new credentials → `cdk bootstrap aws://<new_account>/<region>` → `cdk deploy`. No code changes.
