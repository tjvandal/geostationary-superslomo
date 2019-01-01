import boto3
import goes16s3

noaas3 = goes16s3.NOAAGOESS3()
year_day_pairs = noaas3.year_day_pairs()

input_file = "emr-input.txt"
with open(input_file, "w") as fopen:
    for year, day in year_day_pairs:
        fopen.write("%04i\t%03i\n" % (year, day)) 
        break

emr = boto3.client('emr', region_name='us-east-1')

resp = emr.run_job_flow(
    Name='slowmo-data',
    ReleaseLabel='emr-5.20.0',
    Instances={
        'InstanceGroups': [
            {'Name': 'master',
             'InstanceRole': 'MASTER',
              'InstanceType': 'c1.medium',
             'InstanceCount': 1},
            {'Name': 'core',
             'InstanceRole': 'CORE',
             'InstanceType': 'c1.medium',
             'InstanceCount': 1}
            ]
    },
    Steps=[
        {'Name': 'Generate SlowMo Training Data',
         'HadoopJarStep': {
            'Args': ['hadoop-streaming',
                     '-files', 'emr_mapper.py',
                     '-mapper', 'python emr_mapper.py',
                     '-input', input_file,
                     '-output', 's3://nex-goes-slowmo/output',
                ],
             'Jar': 'command-runner.jar'
            }
        }],
    JobFlowRole='EMR_EC2_DefaultRole',
    ServiceRole='EMR_DefaultRole'
)
