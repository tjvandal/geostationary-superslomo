import boto3
import goes16s3
import sys

goespytorch = goes16s3.GOESDatasetS3()

for line in sys.stdin:
    year, day = line.strip().split('\t')
    goespytorch.write_example_blocks_to_s3(int(year),int(day))
    print("")
