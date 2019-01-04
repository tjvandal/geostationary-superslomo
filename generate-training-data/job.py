import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
import boto3
import goes16s3

BUCKET_NAME = 'nex-goes-slowmo'

noaas3 = goes16s3.NOAAGOESS3()
year_day_pairs = noaas3.year_day_pairs()

input_file = "emr-input.txt"
with open(input_file, "w") as fopen:
    for i, (year, day) in enumerate(year_day_pairs):
        fopen.write("%04i\t%03i\n" % (year, day)) 
        if i > 1:
             break

cmd = ['tar', '-cvf', 'emr-goes-dependencies.tar.gz', '../goes16s3.py', '../utils.py', 'mapper.py']
subprocess.call(cmd)

s3 = boto3.resource('s3', region_name='us-east-1')
bucket = s3.Bucket(BUCKET_NAME)
bucket.upload_file('emr-goes-dependencies.tar.gz', 'emr-goes-dependencies.tar.gz')
bucket.upload_file(input_file, input_file)
bucket.upload_file('emr_configs.sh', 'emr_configs.sh')
bucket.upload_file('mapper.py', 'mapper.py')

emr = boto3.client('emr', region_name='us-east-1')
resp = emr.run_job_flow(
    Name='slomo-data',
    LogUri='s3://%s/logs' % BUCKET_NAME,
    ReleaseLabel='emr-5.20.0',
    Instances={
        'InstanceGroups': [
            {'Name': 'master',
             'Market': 'ON_DEMAND',
             'InstanceRole': 'MASTER',
             'InstanceType': 'c5.xlarge',
             'InstanceCount': 1,
	     'EbsConfiguration' : {
		     'EbsBlockDeviceConfigs': [
			{
			   'VolumeSpecification':
			    {
			     'VolumeType': 'gp2',
			     'SizeInGB': 16
			    },
			}
			],
	      },
	    },
            {'Name': 'core',
             'InstanceRole': 'CORE',
             'InstanceType': 'c5.xlarge',
             'InstanceCount': 1,       
	     'EbsConfiguration' : {
		     'EbsBlockDeviceConfigs': [
			{
			   'VolumeSpecification':
			    {
			     'VolumeType': 'gp2',
			     'SizeInGB': 16
			    },
			}
			],
	      },

	    }
        ],
	'KeepJobFlowAliveWhenNoSteps': True,
	'TerminationProtected': False,
	'Ec2KeyName': 'nex-personal',
    },
    Steps=[
        {'Name': 'Generate SlowMo Training Data',
	 'ActionOnFailure': 'CANCEL_AND_WAIT',
         'HadoopJarStep': {
            'Args': ['hadoop-streaming',
                     '-files',  's3://%s/mapper.py' % (BUCKET_NAME),
                     '-mapper', 'python mapper.py',
                     #'-reducer','python reducer.py',
                     '-input',  's3://%s/%s' % (BUCKET_NAME, input_file),
                     '-output', 's3://%s/output' % BUCKET_NAME,
                    ],
             'Jar': 'command-runner.jar'
            }
        }],
    JobFlowRole='EMR_EC2_DefaultRole',
    ServiceRole='EMR_DefaultRole',
    BootstrapActions=[
         {
            'Name': 'python-env',
            'ScriptBootstrapAction': {
	        'Path': 's3://%s/emr_configs.sh' % BUCKET_NAME,
	     }
          }
	]
)
