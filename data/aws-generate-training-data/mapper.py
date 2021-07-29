#!/usr/bin/python

import boto3
import os
import sys
sys.path.append('/home/hadoop/src')
import goes16s3

goespytorch = goes16s3.GOESDatasetS3()

for line in sys.stdin:
    line = line.strip()
    year, day = line.split('\t')
    goespytorch.write_example_blocks_to_s3(int(year),int(day))
    #print("Completed year: %i and day %i" 5 (year, day))
    print(int(year), int(day))
