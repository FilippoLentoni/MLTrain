import * as path from 'path';
import * as fs from 'fs';
import * as zlib from 'zlib';
import * as cdk from 'aws-cdk-lib';
import {
  Stack,
  StackProps,
  RemovalPolicy,
  CfnOutput,
  Fn,
  aws_s3 as s3,
  aws_iam as iam,
  aws_s3_deployment as s3deploy,
  aws_sagemaker as sagemaker,
} from 'aws-cdk-lib';
import { Construct } from 'constructs';

// AWS's public ECR for SageMaker built-in images in us-east-2 (the only region we deploy to).
// See https://docs.aws.amazon.com/sagemaker/latest/dg-ecr-paths/sagemaker-algo-docker-registry-paths.html
const DEPLOY_REGION = 'us-east-2';
const SAGEMAKER_ECR = `257758044811.dkr.ecr.${DEPLOY_REGION}.amazonaws.com`;
const XGBOOST_IMAGE = `${SAGEMAKER_ECR}/sagemaker-xgboost:1.7-1`;
const SKLEARN_IMAGE = `${SAGEMAKER_ECR}/sagemaker-scikit-learn:1.2-1-cpu-py3`;
const DEFAULT_PROCESSING_INSTANCE_TYPE = 'ml.t3.medium';
const DEFAULT_TRAINING_INSTANCE_TYPE = 'ml.m5.large';

function writeTarOctal(header: Buffer, offset: number, length: number, value: number) {
  header.write(value.toString(8).padStart(length - 1, '0') + '\0', offset, length, 'ascii');
}

function buildTarHeader(fileName: string, fileSize: number) {
  const header = Buffer.alloc(512);
  header.write(fileName, 0, 100, 'utf8');
  writeTarOctal(header, 100, 8, 0o644);
  writeTarOctal(header, 108, 8, 0);
  writeTarOctal(header, 116, 8, 0);
  writeTarOctal(header, 124, 12, fileSize);
  writeTarOctal(header, 136, 12, 0);
  header.fill(' ', 148, 156);
  header.write('0', 156, 1, 'ascii');
  header.write('ustar', 257, 6, 'ascii');
  header.write('00', 263, 2, 'ascii');

  const checksum = header.reduce((sum, byte) => sum + byte, 0);
  header.write(checksum.toString(8).padStart(6, '0') + '\0 ', 148, 8, 'ascii');
  return header;
}

function writeSourceDirTarball(scriptsDir: string, destination: string) {
  const fileName = 'train.py';
  const fileContents = fs.readFileSync(path.join(scriptsDir, fileName));
  const paddingLength = (512 - (fileContents.length % 512)) % 512;
  const tarContents = Buffer.concat([
    buildTarHeader(fileName, fileContents.length),
    fileContents,
    Buffer.alloc(paddingLength),
    Buffer.alloc(1024),
  ]);

  fs.writeFileSync(destination, zlib.gzipSync(tarContents));
}

export class MLTrainStack extends Stack {
  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    const xgboostImage = XGBOOST_IMAGE;
    const sklearnImage = SKLEARN_IMAGE;

    const bucket = new s3.Bucket(this, 'PipelineBucket', {
      removalPolicy: RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      encryption: s3.BucketEncryption.S3_MANAGED,
      versioned: false,
    });

    const scriptsDir = path.join(__dirname, '..', 'scripts');

    const scriptsDeployment = new s3deploy.BucketDeployment(this, 'ScriptsDeployment', {
      sources: [s3deploy.Source.asset(scriptsDir)],
      destinationBucket: bucket,
      destinationKeyPrefix: 'code',
      retainOnDelete: false,
    });

    // Script-mode training requires the training script(s) in a tar.gz that the
    // XGBoost framework container downloads via `sagemaker_submit_directory`.
    // Build sourcedir.tar.gz at synth time into a dedicated build dir, then upload it.
    const buildDir = path.join(__dirname, '..', '.cdk-build', 'sourcedir');
    fs.mkdirSync(buildDir, { recursive: true });
    writeSourceDirTarball(scriptsDir, path.join(buildDir, 'sourcedir.tar.gz'));

    const sourceDirDeployment = new s3deploy.BucketDeployment(this, 'TrainSourceDirDeployment', {
      sources: [s3deploy.Source.asset(buildDir)],
      destinationBucket: bucket,
      destinationKeyPrefix: 'sourcedir',
      retainOnDelete: false,
    });

    const role = new iam.Role(this, 'SageMakerExecutionRole', {
      assumedBy: new iam.ServicePrincipal('sagemaker.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSageMakerFullAccess'),
      ],
    });
    bucket.grantReadWrite(role);

    const bucketUri = `s3://${bucket.bucketName}`;
    const codePreprocess = `${bucketUri}/code/preprocess.py`;
    const codePostprocess = `${bucketUri}/code/postprocess.py`;
    const trainOut = `${bucketUri}/pipeline/preprocess/train`;
    const valOut = `${bucketUri}/pipeline/preprocess/validation`;
    const testOut = `${bucketUri}/pipeline/preprocess/test`;
    const trainingOut = `${bucketUri}/pipeline/training`;
    const metricsOut = `${bucketUri}/pipeline/postprocess/metrics`;

    const pipelineDef = {
      Version: '2020-12-01',
      Metadata: {},
      Parameters: [
        {
          Name: 'ProcessingInstanceType',
          Type: 'String',
          DefaultValue: DEFAULT_PROCESSING_INSTANCE_TYPE,
        },
        {
          Name: 'TrainingInstanceType',
          Type: 'String',
          DefaultValue: DEFAULT_TRAINING_INSTANCE_TYPE,
        },
      ],
      Steps: [
        {
          Name: 'PreprocessData',
          Type: 'Processing',
          Arguments: {
            RoleArn: role.roleArn,
            ProcessingResources: {
              ClusterConfig: {
                InstanceCount: 1,
                InstanceType: { Get: 'Parameters.ProcessingInstanceType' },
                VolumeSizeInGB: 30,
              },
            },
            AppSpecification: {
              ImageUri: sklearnImage,
              ContainerEntrypoint: [
                'python3',
                '/opt/ml/processing/input/code/preprocess.py',
              ],
            },
            ProcessingInputs: [
              {
                InputName: 'code',
                AppManaged: false,
                S3Input: {
                  S3Uri: codePreprocess,
                  LocalPath: '/opt/ml/processing/input/code',
                  S3DataType: 'S3Prefix',
                  S3InputMode: 'File',
                  S3DataDistributionType: 'FullyReplicated',
                  S3CompressionType: 'None',
                },
              },
            ],
            ProcessingOutputConfig: {
              Outputs: [
                {
                  OutputName: 'train',
                  AppManaged: false,
                  S3Output: {
                    S3Uri: trainOut,
                    LocalPath: '/opt/ml/processing/output/train',
                    S3UploadMode: 'EndOfJob',
                  },
                },
                {
                  OutputName: 'validation',
                  AppManaged: false,
                  S3Output: {
                    S3Uri: valOut,
                    LocalPath: '/opt/ml/processing/output/validation',
                    S3UploadMode: 'EndOfJob',
                  },
                },
                {
                  OutputName: 'test',
                  AppManaged: false,
                  S3Output: {
                    S3Uri: testOut,
                    LocalPath: '/opt/ml/processing/output/test',
                    S3UploadMode: 'EndOfJob',
                  },
                },
              ],
            },
          },
        },
        {
          Name: 'TrainModel',
          Type: 'Training',
          Arguments: {
            RoleArn: role.roleArn,
            AlgorithmSpecification: {
              TrainingImage: xgboostImage,
              TrainingInputMode: 'File',
            },
            ResourceConfig: {
              InstanceCount: 1,
              InstanceType: { Get: 'Parameters.TrainingInstanceType' },
              VolumeSizeInGB: 30,
            },
            StoppingCondition: { MaxRuntimeInSeconds: 3600 },
            OutputDataConfig: { S3OutputPath: trainingOut },
            InputDataConfig: [
              {
                ChannelName: 'train',
                ContentType: 'text/csv',
                DataSource: {
                  S3DataSource: {
                    S3DataType: 'S3Prefix',
                    S3Uri: {
                      Get: "Steps.PreprocessData.ProcessingOutputConfig.Outputs['train'].S3Output.S3Uri",
                    },
                    S3DataDistributionType: 'FullyReplicated',
                  },
                },
              },
              {
                ChannelName: 'validation',
                ContentType: 'text/csv',
                DataSource: {
                  S3DataSource: {
                    S3DataType: 'S3Prefix',
                    S3Uri: {
                      Get: "Steps.PreprocessData.ProcessingOutputConfig.Outputs['validation'].S3Output.S3Uri",
                    },
                    S3DataDistributionType: 'FullyReplicated',
                  },
                },
              },
            ],
            HyperParameters: {
              // Script-mode markers: the XGBoost framework container sees these
              // and runs `python train.py --<hp> <value> ...` instead of the built-in algorithm.
              sagemaker_program: 'train.py',
              sagemaker_submit_directory: `${bucketUri}/sourcedir/sourcedir.tar.gz`,
              sagemaker_region: DEPLOY_REGION,
              sagemaker_container_log_level: '20',
              // Forwarded as CLI args to scripts/train.py
              objective: 'binary:logistic',
              num_round: '100',
              max_depth: '5',
              eta: '0.2',
              subsample: '0.8',
              eval_metric: 'error',
            },
          },
        },
        {
          Name: 'EvaluateModel',
          Type: 'Processing',
          Arguments: {
            RoleArn: role.roleArn,
            ProcessingResources: {
              ClusterConfig: {
                InstanceCount: 1,
                InstanceType: { Get: 'Parameters.ProcessingInstanceType' },
                VolumeSizeInGB: 30,
              },
            },
            AppSpecification: {
              ImageUri: xgboostImage,
              ContainerEntrypoint: [
                'python3',
                '/opt/ml/processing/input/code/postprocess.py',
              ],
            },
            ProcessingInputs: [
              {
                InputName: 'code',
                AppManaged: false,
                S3Input: {
                  S3Uri: codePostprocess,
                  LocalPath: '/opt/ml/processing/input/code',
                  S3DataType: 'S3Prefix',
                  S3InputMode: 'File',
                  S3DataDistributionType: 'FullyReplicated',
                  S3CompressionType: 'None',
                },
              },
              {
                InputName: 'model',
                AppManaged: false,
                S3Input: {
                  S3Uri: { Get: 'Steps.TrainModel.ModelArtifacts.S3ModelArtifacts' },
                  LocalPath: '/opt/ml/processing/input/model',
                  S3DataType: 'S3Prefix',
                  S3InputMode: 'File',
                  S3DataDistributionType: 'FullyReplicated',
                  S3CompressionType: 'None',
                },
              },
              {
                InputName: 'test',
                AppManaged: false,
                S3Input: {
                  S3Uri: {
                    Get: "Steps.PreprocessData.ProcessingOutputConfig.Outputs['test'].S3Output.S3Uri",
                  },
                  LocalPath: '/opt/ml/processing/input/test',
                  S3DataType: 'S3Prefix',
                  S3InputMode: 'File',
                  S3DataDistributionType: 'FullyReplicated',
                  S3CompressionType: 'None',
                },
              },
            ],
            ProcessingOutputConfig: {
              Outputs: [
                {
                  OutputName: 'metrics',
                  AppManaged: false,
                  S3Output: {
                    S3Uri: metricsOut,
                    LocalPath: '/opt/ml/processing/output/metrics',
                    S3UploadMode: 'EndOfJob',
                  },
                },
              ],
            },
          },
        },
      ],
    };

    const pipelineDefinitionBody = this.toJsonString(pipelineDef);

    const cfnPipeline = new sagemaker.CfnPipeline(this, 'XGBoostPipeline', {
      pipelineName: `xgboost-pipeline-${id}`,
      roleArn: role.roleArn,
      pipelineDefinition: { PipelineDefinitionBody: pipelineDefinitionBody },
    });
    cfnPipeline.node.addDependency(scriptsDeployment);
    cfnPipeline.node.addDependency(sourceDirDeployment);

    new CfnOutput(this, 'BucketName', { value: bucket.bucketName });
    new CfnOutput(this, 'PipelineName', { value: cfnPipeline.pipelineName! });
    new CfnOutput(this, 'ExecutionRoleArn', { value: role.roleArn });
    new CfnOutput(this, 'StartCommand', {
      value: Fn.sub(
        'aws sagemaker start-pipeline-execution --pipeline-name ${P} --pipeline-parameters Name=ProcessingInstanceType,Value=${ProcessingInstanceType} Name=TrainingInstanceType,Value=${TrainingInstanceType}',
        {
          P: cfnPipeline.pipelineName!,
          ProcessingInstanceType: DEFAULT_PROCESSING_INSTANCE_TYPE,
          TrainingInstanceType: DEFAULT_TRAINING_INSTANCE_TYPE,
        },
      ),
    });
  }
}

// CDK app entry point — synthesises the stack into CloudFormation when `cdk` runs.
const app = new cdk.App();
new MLTrainStack(app, 'MLTrainStack', {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: DEPLOY_REGION,
  },
});
app.synth();
