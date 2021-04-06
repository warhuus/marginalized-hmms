#%%
import json
import os

import boto3

#%%
with open(os.path.join('..', '..', 'aws.json')) as f:
    access = json.load(f)

ACCESS_KEY = access['access-id-key']
SECRET_ACCESS_KEY = access['secret-access-key']

#%%

s3 = boto3.resource('s3')

#%%