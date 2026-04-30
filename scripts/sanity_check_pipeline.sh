#!/usr/bin/env bash
set -euo pipefail

REGION="${REGION:-us-east-2}"
STACK_NAME="${STACK_NAME:-MLTrainStack}"
PIPELINE_NAME="${PIPELINE_NAME:-xgboost-pipeline-MLTrainStack}"

echo "Checking AWS identity..."
aws sts get-caller-identity --output table

echo
echo "Finding stack bucket..."
BUCKET="$(
  aws cloudformation describe-stacks \
    --region "$REGION" \
    --stack-name "$STACK_NAME" \
    --query "Stacks[0].Outputs[?OutputKey=='BucketName'].OutputValue | [0]" \
    --output text
)"

if [[ -z "$BUCKET" || "$BUCKET" == "None" ]]; then
  echo "Could not find BucketName output on stack $STACK_NAME in $REGION" >&2
  exit 1
fi

echo "Bucket: $BUCKET"

echo
echo "Finding latest succeeded pipeline execution..."
EXEC_ARN="$(
  aws sagemaker list-pipeline-executions \
    --region "$REGION" \
    --pipeline-name "$PIPELINE_NAME" \
    --sort-by CreationTime \
    --sort-order Descending \
    --max-results 20 \
    --query "PipelineExecutionSummaries[?PipelineExecutionStatus=='Succeeded'].PipelineExecutionArn | [0]" \
    --output text
)"

if [[ -z "$EXEC_ARN" || "$EXEC_ARN" == "None" ]]; then
  echo "No succeeded executions found for $PIPELINE_NAME in $REGION" >&2
  exit 1
fi

echo "Execution: $EXEC_ARN"

echo
echo "Pipeline steps:"
aws sagemaker list-pipeline-execution-steps \
  --region "$REGION" \
  --pipeline-execution-arn "$EXEC_ARN" \
  --query "PipelineExecutionSteps[].{Step:StepName,Status:StepStatus,Started:StartTime,Ended:EndTime,Failure:FailureReason}" \
  --output table

STEPS_JSON="$(mktemp)"
aws sagemaker list-pipeline-execution-steps \
  --region "$REGION" \
  --pipeline-execution-arn "$EXEC_ARN" \
  --output json > "$STEPS_JSON"

PREPROCESS_JOB="$(
  aws sagemaker list-pipeline-execution-steps \
    --region "$REGION" \
    --pipeline-execution-arn "$EXEC_ARN" \
    --query "PipelineExecutionSteps[?StepName=='PreprocessData'].Metadata.ProcessingJob.Arn | [0]" \
    --output text | awk -F/ '{print $NF}'
)"

TRAINING_JOB="$(
  aws sagemaker list-pipeline-execution-steps \
    --region "$REGION" \
    --pipeline-execution-arn "$EXEC_ARN" \
    --query "PipelineExecutionSteps[?StepName=='TrainModel'].Metadata.TrainingJob.Arn | [0]" \
    --output text | awk -F/ '{print $NF}'
)"

EVALUATE_JOB="$(
  aws sagemaker list-pipeline-execution-steps \
    --region "$REGION" \
    --pipeline-execution-arn "$EXEC_ARN" \
    --query "PipelineExecutionSteps[?StepName=='EvaluateModel'].Metadata.ProcessingJob.Arn | [0]" \
    --output text | awk -F/ '{print $NF}'
)"

echo
echo "Job names:"
echo "  Preprocess: $PREPROCESS_JOB"
echo "  Training:   $TRAINING_JOB"
echo "  Evaluate:   $EVALUATE_JOB"

echo
echo "Training job resource config:"
aws sagemaker describe-training-job \
  --region "$REGION" \
  --training-job-name "$TRAINING_JOB" \
  --query "{Status:TrainingJobStatus,InstanceType:ResourceConfig.InstanceType,InstanceCount:ResourceConfig.InstanceCount,VolumeGB:ResourceConfig.VolumeSizeInGB,ModelArtifacts:ModelArtifacts.S3ModelArtifacts,TrainingTimeSeconds:TrainingTimeInSeconds}" \
  --output table

MODEL_S3="$(
  aws sagemaker describe-training-job \
    --region "$REGION" \
    --training-job-name "$TRAINING_JOB" \
    --query "ModelArtifacts.S3ModelArtifacts" \
    --output text
)"

echo
echo "Model artifact:"
echo "  $MODEL_S3"
aws s3 ls "$MODEL_S3"

MODEL_TMP="$(mktemp)"
aws s3 cp "$MODEL_S3" "$MODEL_TMP" --quiet
echo "Model archive contents:"
tar -tzf "$MODEL_TMP"

echo
echo "Evaluation metrics:"
METRICS_S3="s3://$BUCKET/pipeline/postprocess/metrics/metrics.json"
echo "  $METRICS_S3"
aws s3 cp "$METRICS_S3" -
echo

echo
echo "Preprocess log sanity lines:"
aws logs filter-log-events \
  --region "$REGION" \
  --log-group-name "/aws/sagemaker/ProcessingJobs" \
  --log-stream-name-prefix "$PREPROCESS_JOB/" \
  --query "events[?contains(message, 'train=')].message" \
  --output text

echo
echo "Training log sanity lines:"
aws logs filter-log-events \
  --region "$REGION" \
  --log-group-name "/aws/sagemaker/TrainingJobs" \
  --log-stream-name-prefix "$TRAINING_JOB/" \
  --query "events[?contains(message, 'train shape') || contains(message, 'val shape') || contains(message, '[99]') || contains(message, 'Model saved')].message" \
  --output text

echo
echo "Evaluation log sanity lines:"
aws logs filter-log-events \
  --region "$REGION" \
  --log-group-name "/aws/sagemaker/ProcessingJobs" \
  --log-stream-name-prefix "$EVALUATE_JOB/" \
  --query "events[?contains(message, 'accuracy=')].message" \
  --output text

rm -f "$STEPS_JSON" "$MODEL_TMP"
