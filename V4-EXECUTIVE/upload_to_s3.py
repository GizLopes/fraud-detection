import boto3

BUCKET_NAME = "agentcore-outcomes-789065179500-us-east-1-an"

FILE_NAME = "analytics_operational_table.csv"

s3 = boto3.client("s3")

s3.upload_file(
    FILE_NAME,
    BUCKET_NAME,
    FILE_NAME
)

print("File uploaded to S3.")